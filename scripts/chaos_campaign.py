#!/usr/bin/env python3
"""The verification campaign the README's numbers come from: N randomized
chaos trials (crashes, partitions, message loss/duplication) against the
simulated cluster, each continuously checked for the Raft safety
properties. tests/test_chaos.py runs 40 of these (parametrized, for fast
CI); this script runs a much larger campaign and reports aggregate
statistics + a figure, mirroring the "run it for real, not just enough to
pass CI" verification precedent set by rocket_flight_sim's own 300-trial
Monte Carlo campaign.

    python3 scripts/chaos_campaign.py --trials 2000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raft.simulate import NetworkConditions, SimulatedCluster

FIVE_NODES = ("n1", "n2", "n3", "n4", "n5")


def run_trial(trial_seed: int) -> dict:
    rng = random.Random(trial_seed)
    conditions = NetworkConditions(
        latency_ms_range=(1.0, 20.0),
        drop_probability=rng.uniform(0.0, 0.3),
        duplicate_probability=rng.uniform(0.0, 0.2),
    )
    cluster = SimulatedCluster(FIVE_NODES, rng=random.Random(trial_seed), conditions=conditions)

    start_ms = cluster.now_ms
    proposed = 0
    committed_commands = set()
    for _ in range(15):
        action = rng.choice(["run", "crash", "restart", "partition", "heal", "propose"])
        if action == "run":
            cluster.run_for(rng.uniform(100.0, 800.0))
        elif action == "crash":
            alive = [n for n in FIVE_NODES if n not in cluster.crashed]
            if len(alive) > 3:
                cluster.crash(rng.choice(alive))
        elif action == "restart":
            if cluster.crashed:
                cluster.restart(rng.choice(list(cluster.crashed)))
        elif action == "partition":
            shuffled = list(FIVE_NODES)
            rng.shuffle(shuffled)
            split = rng.randint(1, 4)
            cluster.partition([set(shuffled[:split]), set(shuffled[split:])])
        elif action == "heal":
            cluster.heal_partition()
        elif action == "propose":
            result = cluster.propose_via_leader(f"cmd-{proposed}")
            if result is not None and result.accepted:
                proposed += 1

    cluster.heal_partition()
    for nid in list(cluster.crashed):
        cluster.restart(nid)

    recovery_start_ms = cluster.now_ms
    recovered = False
    for _ in range(60):
        if cluster.current_leader() is not None:
            recovered = True
            break
        cluster.run_for(500.0)
    recovery_ms = cluster.now_ms - recovery_start_ms

    cluster.check_log_matching_property()
    for nid in FIVE_NODES:
        committed_commands.update(e.command for e in cluster.committed_log_by_node[nid] if e is not None)

    return {
        "trial_seed": trial_seed,
        "recovered": recovered,
        "recovery_ms": recovery_ms,
        "num_terms": len(cluster.leader_history),
        "proposed": proposed,
        "committed_distinct_commands": len(committed_commands),
        "drop_probability": conditions.drop_probability,
        "duplicate_probability": conditions.duplicate_probability,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("results/chaos_campaign.json"))
    args = parser.parse_args()

    started = time.time()
    results = []
    safety_violations = 0
    for seed in range(args.trials):
        try:
            results.append(run_trial(seed))
        except AssertionError as exc:
            safety_violations += 1
            results.append({"trial_seed": seed, "safety_violation": str(exc)})
        if (seed + 1) % max(1, args.trials // 10) == 0:
            elapsed = time.time() - started
            print(f"{seed + 1}/{args.trials} trials ({elapsed:.1f}s elapsed, {safety_violations} safety violations so far)")

    elapsed = time.time() - started
    recovered_count = sum(1 for r in results if r.get("recovered"))
    recovery_times = [r["recovery_ms"] for r in results if r.get("recovered")]

    summary = {
        "trials": args.trials,
        "elapsed_s": elapsed,
        "trials_per_second": args.trials / elapsed,
        "safety_violations": safety_violations,
        "recovered_count": recovered_count,
        "recovery_rate": recovered_count / args.trials,
        "recovery_ms_mean": sum(recovery_times) / len(recovery_times) if recovery_times else None,
        "recovery_ms_max": max(recovery_times) if recovery_times else None,
        "recovery_ms_p95": sorted(recovery_times)[int(0.95 * len(recovery_times))] if recovery_times else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "trials": results}, f, indent=2)

    print(f"\n{args.trials} trials in {elapsed:.1f}s ({summary['trials_per_second']:.0f} trials/s)")
    print(f"Safety violations: {safety_violations}")
    print(f"Recovered a leader: {recovered_count}/{args.trials} ({summary['recovery_rate']:.1%})")
    if recovery_times:
        print(f"Recovery time: mean={summary['recovery_ms_mean']:.0f}ms p95={summary['recovery_ms_p95']:.0f}ms max={summary['recovery_ms_max']:.0f}ms")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
