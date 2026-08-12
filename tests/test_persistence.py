"""Crash/restart tests focused specifically on §5.6 of the Raft paper:
currentTerm, votedFor, and the log must survive a crash, because a
restarted server that "forgets" any of them can violate safety even
though the in-memory algorithm above it is correct. SimulatedCluster
already persists after every state-mutating call (see `_persist` in
raft/simulate.py) and restart() reconstructs purely from that snapshot --
these tests check the SPECIFIC failure modes durability is protecting
against, not just "does crash+restart work in general" (already covered
broadly by the chaos trials).
"""

from __future__ import annotations

import random

from raft.messages import RequestVoteArgs, RequestVoteReply, Envelope
from raft.node import Role
from raft.simulate import SimulatedCluster

THREE_NODES = ("n1", "n2", "n3")
FIVE_NODES = ("n1", "n2", "n3", "n4", "n5")


def test_restarted_node_remembers_its_vote_and_refuses_a_second_candidate_same_term():
    cluster = SimulatedCluster(THREE_NODES, rng=random.Random(1))
    n1 = cluster.nodes["n1"]

    # n1 votes for n2 in term 1.
    args = RequestVoteArgs(term=1, candidate_id="n2", last_log_index=0, last_log_term=0)
    n1.on_message(Envelope("n2", "n1", args), now_ms=0)
    cluster._persist("n1")
    assert n1.voted_for == "n2"

    cluster.crash("n1")
    cluster.restart("n1")
    restarted = cluster.nodes["n1"]
    assert restarted.current_term == 1
    assert restarted.voted_for == "n2", "votedFor must survive a crash+restart"

    # Now n3 asks for a vote in the SAME term -- must still be refused,
    # even though the in-memory `votes_received`/role state was reset by
    # the restart (a restarted node always rejoins as a follower).
    args2 = RequestVoteArgs(term=1, candidate_id="n3", last_log_index=0, last_log_term=0)
    [reply_env] = restarted.on_message(Envelope("n3", "n1", args2), now_ms=100)
    reply: RequestVoteReply = reply_env.payload
    assert reply.vote_granted is False, "a restarted node must not double-vote in a term it already voted in"


def test_restarted_node_keeps_its_committed_log_entries():
    cluster = SimulatedCluster(THREE_NODES, rng=random.Random(2))
    cluster.run_for(2000.0)
    leader = cluster.current_leader()
    assert leader is not None
    cluster.propose_via_leader("durable-entry")
    cluster.run_for(1000.0)

    follower = next(n for n in THREE_NODES if n != leader)
    entries_before = list(cluster.nodes[follower].log.entries)

    cluster.crash(follower)
    cluster.restart(follower)
    entries_after = list(cluster.nodes[follower].log.entries)
    assert entries_after == entries_before, "the log must survive a crash+restart unchanged"
    commands = [e.command for e in entries_after]
    assert "durable-entry" in commands


def test_restarted_node_rejoins_as_follower_with_volatile_state_reset():
    # Volatile state (commit_index, role) is deliberately NOT persisted --
    # a restarted node re-learns it from the current leader's RPCs. This
    # test pins that as intentional behavior, not an oversight: a node
    # that persisted commit_index and later restarted into a stale,
    # too-high commit_index could apply an entry that its restored log
    # doesn't actually agree with the rest of the cluster on yet.
    cluster = SimulatedCluster(THREE_NODES, rng=random.Random(3))
    cluster.run_for(2000.0)
    leader = cluster.current_leader()
    cluster.propose_via_leader("x")
    cluster.run_for(1000.0)
    assert cluster.nodes[leader].commit_index > 0

    cluster.crash(leader)
    cluster.restart(leader)
    restarted = cluster.nodes[leader]
    assert restarted.role == Role.FOLLOWER
    assert restarted.commit_index == 0
    assert restarted.last_applied == 0


def test_total_power_loss_and_restart_never_loses_a_committed_entry():
    # Every node crashes at once (e.g. a datacenter power event), then all
    # restart together. The committed entry must survive, and the cluster
    # must be able to re-elect a leader and keep serving afterward -- this
    # is only safe BECAUSE persistence covers exactly currentTerm,
    # votedFor, and the log (§5.6), nothing more and nothing less.
    cluster = SimulatedCluster(FIVE_NODES, rng=random.Random(4))
    cluster.run_for(2000.0)
    cluster.propose_via_leader("before-outage")
    cluster.run_for(1000.0)

    for nid in FIVE_NODES:
        cluster.crash(nid)
    for nid in FIVE_NODES:
        cluster.restart(nid)

    for _ in range(60):
        if cluster.current_leader() is not None:
            break
        cluster.run_for(500.0)
    assert cluster.current_leader() is not None, "cluster must recover after total power loss"

    cluster.propose_via_leader("after-outage")
    cluster.run_for(2000.0)
    for nid in FIVE_NODES:
        commands = [e.command for e in cluster.committed_log_by_node[nid] if e is not None]
        assert "before-outage" in commands, f"{nid} lost a committed entry across total power loss"
    cluster.check_log_matching_property()
