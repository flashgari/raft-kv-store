"""KV store correctness against the simulated cluster: basic operations,
dedup of retried commands, and survival of a leader crash mid-request."""

from __future__ import annotations

import random

from raft.kvstore import ClientCommand, ConcurrentWorkloadDriver, KVCluster, KVStateMachine, Reply, SimulatedKVClient
from raft.messages import LogEntry
from raft.simulate import SimulatedCluster

THREE_NODES = ("n1", "n2", "n3")
FIVE_NODES = ("n1", "n2", "n3", "n4", "n5")


def _ready_cluster(node_ids, seed) -> KVCluster:
    cluster = SimulatedCluster(node_ids, rng=random.Random(seed))
    kv = KVCluster(cluster)
    cluster.run_for(2000.0)
    assert cluster.current_leader() is not None
    return kv


# ---------------------------------------------------------------------
# KVStateMachine unit behavior (no cluster needed)
# ---------------------------------------------------------------------


def test_put_then_get_round_trips():
    sm = KVStateMachine()
    sm.apply(LogEntry(1, 1, ClientCommand("c1", 1, "put", "x", "1")))
    reply = sm.apply(LogEntry(1, 2, ClientCommand("c1", 2, "get", "x", None)))
    assert reply == Reply(ok=True, value="1")


def test_get_of_missing_key_reports_not_ok():
    sm = KVStateMachine()
    reply = sm.apply(LogEntry(1, 1, ClientCommand("c1", 1, "get", "missing", None)))
    assert reply.ok is False


def test_delete_reports_whether_the_key_existed():
    sm = KVStateMachine()
    sm.apply(LogEntry(1, 1, ClientCommand("c1", 1, "put", "x", "1")))
    r1 = sm.apply(LogEntry(1, 2, ClientCommand("c1", 2, "delete", "x", None)))
    assert r1.existed is True
    r2 = sm.apply(LogEntry(1, 3, ClientCommand("c1", 3, "delete", "x", None)))
    assert r2.existed is False


def test_duplicate_sequence_number_returns_cached_reply_without_reapplying():
    sm = KVStateMachine()
    sm.apply(LogEntry(1, 1, ClientCommand("c1", 1, "put", "x", "1")))
    assert sm.applied_count == 1
    # Same client, same seq, DIFFERENT value -- simulates a retried RPC
    # whose original request actually already landed. Must be ignored.
    replayed = sm.apply(LogEntry(1, 2, ClientCommand("c1", 1, "put", "x", "DIFFERENT")))
    assert sm.applied_count == 1, "a duplicate (client_id, seq) must not be re-applied"
    assert replayed.value == "1"
    assert sm.store["x"] == "1"


def test_a_leader_noop_entry_is_ignored_by_the_state_machine():
    from raft.node import NOOP

    sm = KVStateMachine()
    result = sm.apply(LogEntry(1, 1, NOOP))
    assert result is None
    assert sm.applied_count == 0


# ---------------------------------------------------------------------
# End to end against the simulated cluster
# ---------------------------------------------------------------------


def test_put_is_visible_to_a_get_from_any_node_after_committing():
    kv = _ready_cluster(THREE_NODES, seed=1)
    client = SimulatedKVClient(kv, "client-a")
    put_reply = client.put("x", "42")
    assert put_reply.ok is True
    get_reply = client.get("x")
    assert get_reply == Reply(ok=True, value="42")
    # And every node's OWN state machine (not just the leader's) agrees.
    for nid in THREE_NODES:
        assert kv.state_machines[nid].store.get("x") == "42"


def test_client_survives_a_leader_crash_mid_session():
    kv = _ready_cluster(FIVE_NODES, seed=2)
    client = SimulatedKVClient(kv, "client-b")
    client.put("a", "1")

    leader = kv.cluster.current_leader()
    kv.cluster.crash(leader)

    # This call spans the leadership change: the client doesn't know it
    # crashed until its in-flight request stalls and it re-discovers the
    # new leader.
    reply = client.put("b", "2")
    assert reply.ok is True
    new_leader = kv.cluster.current_leader()
    assert new_leader is not None and new_leader != leader
    assert kv.state_machines[new_leader].store.get("b") == "2"


def test_delete_then_get_reports_key_absent():
    kv = _ready_cluster(THREE_NODES, seed=3)
    client = SimulatedKVClient(kv, "client-c")
    client.put("k", "v")
    client.delete("k")
    reply = client.get("k")
    assert reply.ok is False


def test_driver_reuses_the_same_client_across_multiple_run_calls():
    # Regression test for a real bug: ConcurrentWorkloadDriver originally
    # constructed a brand new SimulatedKVClient (seq starting at 0) on
    # EVERY call to run(). Calling run() twice for the same client_id then
    # reissued seq=1 for a genuinely new command, which KVStateMachine's
    # own (client_id, seq) dedup logic treated as a replay of the FIRST
    # command -- the second command was silently dropped and the client
    # got back the first command's stale cached reply. Caught by
    # test_linearizability.py's leader-crash test coming back
    # non-linearizable; this test pins the fix directly against the
    # driver/state-machine without needing the full checker.
    kv = _ready_cluster(THREE_NODES, seed=4)
    driver = ConcurrentWorkloadDriver(kv, step_ms=10.0)

    history_1 = driver.run({"c1": [("put", "x", "first")]}, max_wait_ms=5000.0)
    assert history_1[0].reply == Reply(ok=True, value="first")

    history_2 = driver.run({"c1": [("put", "x", "second")]}, max_wait_ms=5000.0)
    assert history_2[0].reply == Reply(ok=True, value="second"), (
        "second run() call for the same client_id was treated as a duplicate "
        "of the first -- client sequence numbers did not survive across run() calls"
    )
    leader = kv.cluster.current_leader()
    assert kv.state_machines[leader].store["x"] == "second"
