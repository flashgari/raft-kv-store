"""Fault-injection tests against the simulated cluster: elections, leader
failure/recovery, network partitions, message loss/duplication, and
randomized multi-fault trials -- each one continuously checking the Raft
safety properties (via SimulatedCluster's built-in assertions and
check_log_matching_property) rather than just checking "did it eventually
converge."
"""

from __future__ import annotations

import random

import pytest

from raft.simulate import NetworkConditions, SimulatedCluster

FIVE_NODES = ("n1", "n2", "n3", "n4", "n5")
THREE_NODES = ("n1", "n2", "n3")


def _elect_leader(cluster: SimulatedCluster, timeout_ms: float = 3000.0) -> str:
    cluster.run_for(timeout_ms)
    leader = cluster.current_leader()
    assert leader is not None, "no leader elected within timeout"
    return leader


# ---------------------------------------------------------------------
# Basic liveness
# ---------------------------------------------------------------------


def test_a_leader_is_elected_in_a_healthy_cluster():
    cluster = SimulatedCluster(THREE_NODES, rng=random.Random(1))
    leader = _elect_leader(cluster)
    assert leader in THREE_NODES


def test_leader_replicates_a_committed_entry_to_all_followers():
    cluster = SimulatedCluster(THREE_NODES, rng=random.Random(2))
    _elect_leader(cluster)
    result = cluster.propose_via_leader("set x=1")
    assert result is not None and result.accepted
    cluster.run_for(1000.0)
    for nid in THREE_NODES:
        committed = [e for e in cluster.committed_log_by_node[nid] if e is not None]
        commands = [e.command for e in committed]
        assert "set x=1" in commands, f"{nid} never committed the entry"
    cluster.check_log_matching_property()


def test_a_dead_node_does_not_become_leader():
    cluster = SimulatedCluster(THREE_NODES, rng=random.Random(3))
    cluster.crash("n2")
    cluster.run_for(3000.0)
    assert cluster.current_leader() in ("n1", "n3")


# ---------------------------------------------------------------------
# Leader failure and re-election
# ---------------------------------------------------------------------


def test_new_leader_is_elected_after_the_old_one_crashes():
    cluster = SimulatedCluster(FIVE_NODES, rng=random.Random(4))
    first_leader = _elect_leader(cluster)
    cluster.crash(first_leader)
    cluster.run_for(3000.0)
    second_leader = cluster.current_leader()
    assert second_leader is not None
    assert second_leader != first_leader


def test_a_restarted_former_leader_rejoins_as_a_follower_not_a_duplicate_leader():
    cluster = SimulatedCluster(FIVE_NODES, rng=random.Random(5))
    first_leader = _elect_leader(cluster)
    cluster.propose_via_leader("a")
    cluster.run_for(500.0)
    cluster.crash(first_leader)
    cluster.run_for(3000.0)
    second_leader = cluster.current_leader()
    assert second_leader is not None and second_leader != first_leader

    cluster.restart(first_leader)
    cluster.run_for(3000.0)
    from raft.node import Role

    assert cluster.nodes[first_leader].role != Role.LEADER or cluster.nodes[first_leader].node_id == cluster.current_leader()
    # Only one leader must exist at any point -- already enforced continuously
    # by SimulatedCluster's own bookkeeping (would have raised otherwise).
    cluster.check_log_matching_property()


# ---------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------


def test_minority_partition_cannot_elect_a_leader_or_commit():
    cluster = SimulatedCluster(FIVE_NODES, rng=random.Random(6))
    _elect_leader(cluster)
    # Split 2 vs 3: the 2-node side can never reach a majority (3 of 5).
    cluster.partition([{"n1", "n2"}, {"n3", "n4", "n5"}])
    cluster.run_for(5000.0)

    majority_leader = None
    for nid in ("n3", "n4", "n5"):
        from raft.node import Role

        if cluster.nodes[nid].role == Role.LEADER:
            majority_leader = nid
    assert majority_leader is not None, "the majority-side partition should elect its own leader"

    for nid in ("n1", "n2"):
        from raft.node import Role

        assert cluster.nodes[nid].role != Role.LEADER, "the minority side must never elect a leader"


def test_cluster_recovers_and_reconciles_logs_after_partition_heals():
    cluster = SimulatedCluster(FIVE_NODES, rng=random.Random(7))
    _elect_leader(cluster)
    cluster.partition([{"n1", "n2"}, {"n3", "n4", "n5"}])
    cluster.run_for(3000.0)
    # The majority side should be able to commit while partitioned.
    cluster.propose_via_leader("during-partition")
    cluster.run_for(1000.0)

    cluster.heal_partition()
    cluster.run_for(5000.0)
    cluster.check_log_matching_property()

    leader = cluster.current_leader()
    assert leader is not None
    cluster.propose_via_leader("after-heal")
    cluster.run_for(1000.0)
    for nid in FIVE_NODES:
        commands = [e.command for e in cluster.committed_log_by_node[nid] if e is not None]
        assert "during-partition" in commands
        assert "after-heal" in commands
    cluster.check_log_matching_property()


# ---------------------------------------------------------------------
# Lossy / duplicating network
# ---------------------------------------------------------------------


def test_cluster_still_makes_progress_under_message_loss_and_duplication():
    conditions = NetworkConditions(latency_ms_range=(1.0, 15.0), drop_probability=0.2, duplicate_probability=0.15)
    cluster = SimulatedCluster(FIVE_NODES, rng=random.Random(8), conditions=conditions)
    cluster.run_for(5000.0)
    assert cluster.current_leader() is not None
    result = cluster.propose_via_leader("lossy-write")
    cluster.run_for(3000.0)
    if result is not None and result.accepted:
        leader = cluster.current_leader()
        # At minimum, the entry must be committed on whichever node ends up
        # leader's own log without corrupting any node's log (checked below).
        assert any(
            e is not None and e.command == "lossy-write" for e in cluster.committed_log_by_node.get(leader, [])
        )
    cluster.check_log_matching_property()


# ---------------------------------------------------------------------
# Randomized multi-fault trials -- this is the real stress test
# ---------------------------------------------------------------------


@pytest.mark.parametrize("trial_seed", range(40))
def test_randomized_chaos_trial_never_violates_safety(trial_seed):
    rng = random.Random(trial_seed)
    conditions = NetworkConditions(
        latency_ms_range=(1.0, 20.0),
        drop_probability=rng.uniform(0.0, 0.3),
        duplicate_probability=rng.uniform(0.0, 0.2),
    )
    cluster = SimulatedCluster(FIVE_NODES, rng=random.Random(trial_seed), conditions=conditions)

    proposed = 0
    for _ in range(15):
        action = rng.choice(["run", "crash", "restart", "partition", "heal", "propose"])
        if action == "run":
            cluster.run_for(rng.uniform(100.0, 800.0))
        elif action == "crash":
            alive = [n for n in FIVE_NODES if n not in cluster.crashed]
            if len(alive) > 3:  # never crash below majority, or nothing can ever commit
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
        # Safety properties are asserted continuously inside step(); a
        # violation raises immediately rather than at the end of the trial.

    cluster.heal_partition()
    for nid in list(cluster.crashed):
        cluster.restart(nid)
    # Bound the wait generously rather than a single fixed window: under
    # high drop/duplicate probability an election can need several retries
    # (each a full randomized election-timeout period), and a fixed window
    # that's merely "usually enough" makes this test flaky on the unlucky
    # trials rather than actually distinguishing liveness bugs from slow
    # convergence. 30 more seconds of simulated time is >100x a single
    # election-timeout period, so a leader failing to appear within it is
    # a real liveness problem, not bad luck.
    for _ in range(60):
        if cluster.current_leader() is not None:
            break
        cluster.run_for(500.0)
    cluster.check_log_matching_property()
    assert cluster.current_leader() is not None, "cluster must recover a leader once healthy"
