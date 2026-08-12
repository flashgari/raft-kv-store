#!/usr/bin/env python3
"""Live-cluster benchmark: spins up the same real 3-process cluster
run_demo.py does, then measures (1) sequential PUT latency -- the cost of
one full consensus round trip per write, this project's actual unit of
work -- and (2) throughput under N concurrent clients hammering PUTs at
once. Numbers are honestly what they are for a JSON-over-TCP, single-
leader-bottleneck, un-pipelined implementation (see README's "Model
boundaries" for what a production system would change and why).

    python3 scripts/benchmark.py --sequential-ops 200 --concurrent-clients 8 --concurrent-seconds 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
NODE_IDS = ["n1", "n2", "n3"]
PEER_PORTS = {"n1": 9501, "n2": 9502, "n3": 9503}
CLIENT_PORTS = {"n1": 9601, "n2": 9602, "n3": 9603}
DATA_DIR = ROOT / "data" / "benchmark"
CLIENT_ADDRS = [f"127.0.0.1:{CLIENT_PORTS[n]}" for n in NODE_IDS]

from scripts.kv_client import request as kv_request  # noqa: E402


def peers_spec() -> str:
    return ",".join(f"{n}=127.0.0.1:{PEER_PORTS[n]}" for n in NODE_IDS)


def start_node(node_id: str) -> subprocess.Popen:
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_node.py"),
        "--id", node_id, "--port", str(PEER_PORTS[node_id]), "--client-port", str(CLIENT_PORTS[node_id]),
        "--peers", peers_spec(), "--data-dir", str(DATA_DIR),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    idx = min(len(s) - 1, int(p * len(s)))
    return s[idx]


async def sequential_benchmark(n_ops: int) -> dict:
    latencies_ms = []
    for i in range(n_ops):
        t0 = time.perf_counter()
        reply = await kv_request(CLIENT_ADDRS, "put", f"seq-key-{i % 20}", f"v{i}")
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        assert reply.get("ok") is True, f"benchmark PUT failed: {reply}"
    return {
        "n_ops": n_ops,
        "mean_ms": sum(latencies_ms) / len(latencies_ms),
        "p50_ms": percentile(latencies_ms, 0.50),
        "p95_ms": percentile(latencies_ms, 0.95),
        "p99_ms": percentile(latencies_ms, 0.99),
        "max_ms": max(latencies_ms),
        "raw_ms": latencies_ms,
    }


async def concurrent_worker(worker_id: int, deadline: float, keys_per_worker: int) -> int:
    completed = 0
    i = 0
    while time.perf_counter() < deadline:
        reply = await kv_request(CLIENT_ADDRS, "put", f"conc-{worker_id}-{i % keys_per_worker}", f"v{i}")
        if reply.get("ok"):
            completed += 1
        i += 1
    return completed


async def concurrency_sweep(levels: list[int], seconds: float) -> list[dict]:
    results = []
    for n in levels:
        deadline = time.perf_counter() + seconds
        started = time.perf_counter()
        per_client = await asyncio.gather(*[concurrent_worker(w, deadline, keys_per_worker=10) for w in range(n)])
        elapsed = time.perf_counter() - started
        total_ops = sum(per_client)
        results.append({"n_clients": n, "seconds": elapsed, "total_ops": total_ops, "ops_per_second": total_ops / elapsed})
        print(f"  {n:>3} clients: {total_ops:>5} ops in {elapsed:.2f}s = {total_ops / elapsed:6.1f} ops/s")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequential-ops", type=int, default=200)
    parser.add_argument("--concurrency-levels", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--concurrent-seconds", type=float, default=4.0)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args()

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    print("Starting 3 node processes ...")
    procs = {nid: start_node(nid) for nid in NODE_IDS}
    time.sleep(2.5)

    try:
        print(f"Sequential benchmark: {args.sequential_ops} PUTs, one at a time (measures single-write consensus round-trip latency) ...")
        seq = asyncio.run(sequential_benchmark(args.sequential_ops))
        print(f"  mean={seq['mean_ms']:.2f}ms p50={seq['p50_ms']:.2f}ms p95={seq['p95_ms']:.2f}ms p99={seq['p99_ms']:.2f}ms max={seq['max_ms']:.2f}ms")

        print(f"Concurrency sweep: {args.concurrency_levels} clients, {args.concurrent_seconds}s each (measures throughput vs. concurrent load) ...")
        sweep = asyncio.run(concurrency_sweep(args.concurrency_levels, args.concurrent_seconds))

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"sequential": seq, "concurrency_sweep": sweep}, f, indent=2)
        print(f"Wrote {args.output}")
    finally:
        for p in procs.values():
            p.kill()
        for p in procs.values():
            p.wait()
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)


if __name__ == "__main__":
    main()
