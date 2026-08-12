"""Two layers of testing here, deliberately kept separate:

1. Checker self-tests (`test_checker_*`): hand-crafted histories with a
   known correct answer, checked WITHOUT touching the Raft cluster at
   all -- this proves the checker itself is trustworthy (both that it
   accepts a legally-overlapping history and that it REJECTS a real
   violation) before anything downstream relies on it.
2. System tests (`test_kv_cluster_*`): the actual KV cluster, driven by
   several genuinely concurrent clients (via ConcurrentWorkloadDriver),
   checked with the now-trusted checker -- including under network
   chaos, which is the whole point of building this: proving the system
   behaves as a single linearizable register even while nodes crash,
   partitions happen, and messages get lost or duplicated underneath it.
"""

from __future__ import annotations

import random

from raft.kvstore import ClientCommand, ConcurrentWorkloadDriver, KVCluster, Operation, Reply
from raft.linearizability import check_linearizable
from raft.simulate import NetworkConditions, SimulatedCluster

THREE_NODES = ("n1", "n2", "n3")
FIVE_NODES = ("n1", "n2", "n3", "n4", "n5")


# ---------------------------------------------------------------------
# Checker self-tests
# ---------------------------------------------------------------------


def test_checker_accepts_a_purely_sequential_history():
    history = [
        Operation("A", "put", "k", "1", start_ms=0, end_ms=10, reply=Reply(ok=True, value="1")),
        Operation("A", "get", "k", None, start_ms=20, end_ms=30, reply=Reply(ok=True, value="1")),
    ]
    result = check_linearizable(history)
    assert result.linearizable is True


def test_checker_accepts_a_legal_overlapping_ordering():
    # X's put and Y's get OVERLAP in time, so Y observing "not found" is
    # legal (the put's linearization point can be placed after Y's get).
    # Z's get starts strictly after X's put ends, so real-time order
    # forces Z to observe the write.
    history = [
        Operation("X", "put", "k", "1", start_ms=0, end_ms=100, reply=Reply(ok=True, value="1")),
        Operation("Y", "get", "k", None, start_ms=50, end_ms=150, reply=Reply(ok=False, value=None)),
        Operation("Z", "get", "k", None, start_ms=160, end_ms=200, reply=Reply(ok=True, value="1")),
    ]
    result = check_linearizable(history)
    assert result.linearizable is True
    # X must come before Z in the witness order (Y can land anywhere
    # legal relative to X since they overlap).
    order = result.witness_order
    assert order.index(0) < order.index(2)


def test_checker_rejects_a_stale_read_that_violates_real_time_order():
    # Z's get starts AFTER X's put has already returned -- real-time order
    # requires X to linearize before Z, so Z observing "not found" is a
    # genuine violation, not just an unlucky interleaving.
    history = [
        Operation("X", "put", "k", "1", start_ms=0, end_ms=100, reply=Reply(ok=True, value="1")),
        Operation("Z", "get", "k", None, start_ms=110, end_ms=150, reply=Reply(ok=False, value=None)),
    ]
    result = check_linearizable(history)
    assert result.linearizable is False


def test_checker_rejects_a_read_of_a_value_that_was_never_written():
    history = [
        Operation("A", "get", "k", None, start_ms=0, end_ms=10, reply=Reply(ok=True, value="ghost")),
    ]
    result = check_linearizable(history)
    assert result.linearizable is False


def test_checker_accepts_concurrent_non_conflicting_key_operations():
    # Different keys never constrain each other regardless of overlap.
    history = [
        Operation("A", "put", "k1", "1", start_ms=0, end_ms=100, reply=Reply(ok=True, value="1")),
        Operation("B", "put", "k2", "2", start_ms=10, end_ms=90, reply=Reply(ok=True, value="2")),
        Operation("A", "get", "k2", None, start_ms=95, end_ms=120, reply=Reply(ok=True, value="2")),
    ]
    result = check_linearizable(history)
    assert result.linearizable is True


# ---------------------------------------------------------------------
# System tests: a real concurrent workload against the simulated cluster
# ---------------------------------------------------------------------


def _workload(client_prefix: str, n_clients: int, n_ops: int, keys: tuple[str, ...], seed: int):
    rng = random.Random(seed)
    workloads = {}
    for c in range(n_clients):
        cid = f"{client_prefix}-{c}"
        ops = []
        for _ in range(n_ops):
            key = rng.choice(keys)
            kind = rng.choice(["put", "put", "get", "delete"])
            value = f"v{rng.randint(0, 999)}" if kind == "put" else None
            ops.append((kind, key, value))
        workloads[cid] = ops
    return workloads


def test_kv_cluster_is_linearizable_under_concurrent_load_no_faults():
    cluster = SimulatedCluster(THREE_NODES, rng=random.Random(10))
    kv = KVCluster(cluster)
    cluster.run_for(2000.0)
    driver = ConcurrentWorkloadDriver(kv, step_ms=10.0)
    workloads = _workload("c", n_clients=4, n_ops=3, keys=("a", "b"), seed=10)
    history = driver.run(workloads)
    assert len(history) == 12
    result = check_linearizable(history)
    assert result.linearizable is True, f"non-linearizable history! states_explored={result.states_explored}"


def test_kv_cluster_is_linearizable_under_lossy_network():
    conditions = NetworkConditions(latency_ms_range=(1.0, 15.0), drop_probability=0.15, duplicate_probability=0.1)
    cluster = SimulatedCluster(THREE_NODES, rng=random.Random(11), conditions=conditions)
    kv = KVCluster(cluster)
    cluster.run_for(2000.0)
    driver = ConcurrentWorkloadDriver(kv, step_ms=10.0)
    workloads = _workload("c", n_clients=3, n_ops=3, keys=("a",), seed=11)
    history = driver.run(workloads, max_wait_ms=30000.0)
    result = check_linearizable(history)
    assert result.linearizable is True


def test_kv_cluster_is_linearizable_across_a_leader_crash_mid_workload():
    cluster = SimulatedCluster(FIVE_NODES, rng=random.Random(12))
    kv = KVCluster(cluster)
    cluster.run_for(2000.0)
    driver = ConcurrentWorkloadDriver(kv, step_ms=10.0)

    workloads = _workload("c", n_clients=4, n_ops=2, keys=("a", "b"), seed=12)
    # Crash the leader partway through by running the driver in two
    # halves and crashing between them.
    first_half = {cid: ops[:1] for cid, ops in workloads.items()}
    second_half = {cid: ops[1:] for cid, ops in workloads.items()}

    history = driver.run(first_half, max_wait_ms=10000.0)
    leader = cluster.current_leader()
    assert leader is not None
    cluster.crash(leader)
    cluster.run_for(2000.0)  # let a new leader get elected before resuming traffic
    history += driver.run(second_half, max_wait_ms=15000.0)

    result = check_linearizable(history)
    assert result.linearizable is True, f"non-linearizable across leader crash! states_explored={result.states_explored}"
