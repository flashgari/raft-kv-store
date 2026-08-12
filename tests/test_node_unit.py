"""Direct, hand-scripted tests against RaftNode.on_message/tick -- no
simulated network here, just constructing envelopes and checking the
exact reply/state-transition logic against the Raft paper's Figure 2
rules. Chaos/fault-injection tests (partitions, crashes) live in
test_chaos.py against the full simulated cluster.
"""

from __future__ import annotations

import random

from raft.log import RaftLog
from raft.messages import AppendEntriesArgs, Envelope, LogEntry, RequestVoteArgs, RequestVoteReply, AppendEntriesReply
from raft.node import NOOP, RaftNode, Role


def make_node(node_id="n1", peers=("n2", "n3"), seed=1) -> RaftNode:
    return RaftNode(node_id=node_id, peers=peers, rng=random.Random(seed))


# ---------------------------------------------------------------------
# RequestVote
# ---------------------------------------------------------------------


def test_grants_vote_to_candidate_with_up_to_date_log_in_new_term():
    node = make_node()
    args = RequestVoteArgs(term=1, candidate_id="n2", last_log_index=0, last_log_term=0)
    [reply_env] = node.on_message(Envelope("n2", "n1", args), now_ms=0)
    reply: RequestVoteReply = reply_env.payload
    assert reply.vote_granted is True
    assert reply.term == 1
    assert node.voted_for == "n2"
    assert node.current_term == 1


def test_rejects_vote_request_with_stale_term():
    node = make_node()
    node.current_term = 5
    args = RequestVoteArgs(term=3, candidate_id="n2", last_log_index=0, last_log_term=0)
    [reply_env] = node.on_message(Envelope("n2", "n1", args), now_ms=0)
    reply: RequestVoteReply = reply_env.payload
    assert reply.vote_granted is False
    assert reply.term == 5  # tells the stale candidate the real term


def test_refuses_second_vote_in_same_term_to_a_different_candidate():
    node = make_node()
    first = RequestVoteArgs(term=1, candidate_id="n2", last_log_index=0, last_log_term=0)
    node.on_message(Envelope("n2", "n1", first), now_ms=0)
    assert node.voted_for == "n2"

    second = RequestVoteArgs(term=1, candidate_id="n3", last_log_index=0, last_log_term=0)
    [reply_env] = node.on_message(Envelope("n3", "n1", second), now_ms=1)
    assert reply_env.payload.vote_granted is False


def test_regrants_same_candidate_a_duplicate_vote_request_idempotently():
    # A retried/duplicated RequestVote from the SAME candidate in the same
    # term must still be granted (voted_for in (None, candidate_id)) --
    # otherwise a dropped-then-retried vote request could never succeed.
    node = make_node()
    args = RequestVoteArgs(term=1, candidate_id="n2", last_log_index=0, last_log_term=0)
    node.on_message(Envelope("n2", "n1", args), now_ms=0)
    [reply_env] = node.on_message(Envelope("n2", "n1", args), now_ms=1)
    assert reply_env.payload.vote_granted is True


def test_rejects_candidate_with_less_up_to_date_log():
    node = make_node()
    node.log = RaftLog(entries=[LogEntry(0, 0, None), LogEntry(term=2, index=1, command="x")])
    node.current_term = 2
    # Candidate's last_log_term (1) is behind the voter's last_log_term (2).
    args = RequestVoteArgs(term=3, candidate_id="n2", last_log_index=5, last_log_term=1)
    [reply_env] = node.on_message(Envelope("n2", "n1", args), now_ms=0)
    assert reply_env.payload.vote_granted is False


def test_higher_term_in_request_vote_converts_leader_to_follower():
    node = make_node()
    node.role = Role.LEADER
    node.current_term = 2
    node.leader_id = "n1"
    args = RequestVoteArgs(term=5, candidate_id="n2", last_log_index=0, last_log_term=0)
    node.on_message(Envelope("n2", "n1", args), now_ms=0)
    assert node.role == Role.FOLLOWER
    assert node.current_term == 5
    assert node.leader_id is None


# ---------------------------------------------------------------------
# Election end to end (3-node cluster, 2 peers scripted directly)
# ---------------------------------------------------------------------


def test_candidate_becomes_leader_after_majority_votes_and_appends_noop():
    node = make_node()
    # tick() only starts an election once now_ms >= election_deadline_ms;
    # force it deterministically instead of relying on the random deadline.
    node.election_deadline_ms = 0
    envelopes = node.tick(now_ms=1)
    assert node.role == Role.CANDIDATE
    assert len(envelopes) == 2
    assert {e.dst for e in envelopes} == {"n2", "n3"}
    assert all(isinstance(e.payload, RequestVoteArgs) for e in envelopes)

    reply1 = RequestVoteReply(term=node.current_term, vote_granted=True, voter_id="n2")
    out = node.on_message(Envelope("n2", "n1", reply1), now_ms=2)
    # self-vote + n2's vote = 2 of 3 = majority -> already leader after just one peer reply
    assert node.role == Role.LEADER
    assert node.log.last_index == 1  # the no-op entry
    assert node.log.entry_at(1).command is NOOP
    assert len(out) == 2  # heartbeat/replication broadcast to both peers


def test_stale_vote_reply_from_a_previous_term_is_ignored():
    node = make_node()
    node.election_deadline_ms = 0
    node.tick(now_ms=1)
    assert node.current_term == 1
    stale_reply = RequestVoteReply(term=0, vote_granted=True, voter_id="n2")
    node.on_message(Envelope("n2", "n1", stale_reply), now_ms=2)
    assert node.role == Role.CANDIDATE  # ignored: reply.term != current_term


# ---------------------------------------------------------------------
# AppendEntries consistency + conflict resolution
# ---------------------------------------------------------------------


def test_append_entries_rejected_when_prev_log_term_mismatches():
    node = make_node()
    node.log = RaftLog(entries=[LogEntry(0, 0, None), LogEntry(term=1, index=1, command="a")])
    node.current_term = 2
    args = AppendEntriesArgs(term=2, leader_id="n2", prev_log_index=1, prev_log_term=99, entries=(), leader_commit=0)
    [reply_env] = node.on_message(Envelope("n2", "n1", args), now_ms=0)
    reply: AppendEntriesReply = reply_env.payload
    assert reply.success is False
    assert reply.conflict_term == 1
    assert reply.conflict_index == 1


def test_append_entries_appends_new_entries_and_advances_commit_index():
    node = make_node()
    node.current_term = 1
    entries = (LogEntry(term=1, index=1, command="a"), LogEntry(term=1, index=2, command="b"))
    args = AppendEntriesArgs(term=1, leader_id="n2", prev_log_index=0, prev_log_term=0, entries=entries, leader_commit=2)
    [reply_env] = node.on_message(Envelope("n2", "n1", args), now_ms=0)
    assert reply_env.payload.success is True
    assert node.log.last_index == 2
    assert node.commit_index == 2
    assert node.leader_id == "n2"


def test_conflicting_entry_is_truncated_not_merged():
    node = make_node()
    node.log = RaftLog(
        entries=[
            LogEntry(0, 0, None),
            LogEntry(term=1, index=1, command="a"),
            LogEntry(term=1, index=2, command="stale-b"),
            LogEntry(term=1, index=3, command="stale-c"),
        ]
    )
    node.current_term = 2
    new_entries = (LogEntry(term=2, index=2, command="fresh-b"),)
    args = AppendEntriesArgs(term=2, leader_id="n2", prev_log_index=1, prev_log_term=1, entries=new_entries, leader_commit=0)
    node.on_message(Envelope("n2", "n1", args), now_ms=0)
    assert node.log.last_index == 2
    assert node.log.entry_at(2).command == "fresh-b"


def test_duplicate_append_entries_is_not_destructive():
    # A delayed, already-applied AppendEntries retried later must NOT
    # truncate entries a subsequent RPC already confirmed as correct.
    node = make_node()
    node.current_term = 1
    e1 = (LogEntry(term=1, index=1, command="a"),)
    node.on_message(Envelope("n2", "n1", AppendEntriesArgs(1, "n2", 0, 0, e1, 0)), now_ms=0)
    e2 = (LogEntry(term=1, index=2, command="b"),)
    node.on_message(Envelope("n2", "n1", AppendEntriesArgs(1, "n2", 1, 1, e2, 0)), now_ms=1)
    assert node.log.last_index == 2

    # Retry (duplicate) of the FIRST rpc arrives late.
    node.on_message(Envelope("n2", "n1", AppendEntriesArgs(1, "n2", 0, 0, e1, 0)), now_ms=2)
    assert node.log.last_index == 2
    assert node.log.entry_at(2).command == "b"


# ---------------------------------------------------------------------
# Leader commit rule: never commit a previous-term entry by count alone
# ---------------------------------------------------------------------


def test_leader_does_not_commit_previous_term_entry_by_replica_count_alone():
    # Reproduces the Raft paper's Figure 8 scenario directly against the
    # commit-advancement logic: an entry from term 1 sits replicated on a
    # majority of match_index values, but the leader's CURRENT term is 2
    # and it has not yet replicated any term-2 entry. commit_index must
    # stay put until a same-term entry is majority-replicated.
    node = make_node("n1", ("n2", "n3", "n4", "n5"))
    node.role = Role.LEADER
    node.current_term = 2
    node.log = RaftLog(entries=[LogEntry(0, 0, None), LogEntry(term=1, index=1, command="a")])
    node.match_index = {"n2": 1, "n3": 1, "n4": 0, "n5": 0}  # 3/5 incl. self replicated index 1 (term 1)
    node._advance_commit_index()
    assert node.commit_index == 0, "must not commit a prior-term entry via replica count alone"

    # Now the leader appends (and replicates) a term-2 entry at index 2;
    # once THAT is majority-replicated, index 1 becomes committed too
    # (as a consequence of the Log Matching property), in the same call.
    node.log.append(LogEntry(term=2, index=2, command="b"))
    node.match_index = {"n2": 2, "n3": 2, "n4": 0, "n5": 0}
    node._advance_commit_index()
    assert node.commit_index == 2


# ---------------------------------------------------------------------
# take_newly_committed bookkeeping
# ---------------------------------------------------------------------


def test_take_newly_committed_returns_entries_in_order_and_advances_last_applied():
    node = make_node()
    node.log = RaftLog(
        entries=[LogEntry(0, 0, None), LogEntry(1, 1, "a"), LogEntry(1, 2, "b"), LogEntry(1, 3, "c")]
    )
    node.commit_index = 2
    got = node.take_newly_committed()
    assert [e.command for e in got] == ["a", "b"]
    assert node.last_applied == 2
    assert node.take_newly_committed() == []  # nothing new yet
    node.commit_index = 3
    got2 = node.take_newly_committed()
    assert [e.command for e in got2] == ["c"]
