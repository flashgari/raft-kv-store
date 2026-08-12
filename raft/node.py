"""The Raft consensus state machine (paper Figure 2 rules), transport-free.

Design: every public method is a pure(-ish) transition function. It reads
`self` state, mutates it, and returns the list of `Envelope`s that should
be sent as a result -- it never calls a socket, a clock, or `sleep()`
itself. The caller (a simulated network in tests, or the real asyncio
transport in `transport/`) owns time and delivery. This is what makes the
protocol's hardest-to-get-right behavior -- split votes, log conflicts,
the §5.4.2 "leader cannot commit entries from previous terms by counting
replicas alone" rule -- unit-testable in isolation and fuzzable by the
thousand without a single real timer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto

from .log import RaftLog, candidate_log_is_up_to_date
from .messages import (
    AppendEntriesArgs,
    AppendEntriesReply,
    Envelope,
    LogEntry,
    RequestVoteArgs,
    RequestVoteReply,
)

NOOP = object()  # sentinel command for the leader's post-election no-op entry


class Role(Enum):
    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


@dataclass
class ProposeResult:
    accepted: bool
    index: int | None
    term: int | None
    leader_hint: str | None
    envelopes: list[Envelope]


@dataclass
class RaftNode:
    node_id: str
    peers: tuple[str, ...]  # other node ids, not including self
    rng: random.Random
    election_timeout_range_ms: tuple[float, float] = (150.0, 300.0)
    heartbeat_interval_ms: float = 50.0

    # --- persistent state (paper Figure 2) ---
    current_term: int = 0
    voted_for: str | None = None
    log: RaftLog = field(default_factory=RaftLog)

    # --- volatile state, all servers ---
    role: Role = Role.FOLLOWER
    commit_index: int = 0
    last_applied: int = 0
    leader_id: str | None = None

    # --- volatile state, leaders only (reinitialized after each election) ---
    next_index: dict[str, int] = field(default_factory=dict)
    match_index: dict[str, int] = field(default_factory=dict)

    # --- volatile, candidates only ---
    votes_received: set[str] = field(default_factory=set)

    # --- timers (simulated-time ms; caller drives via tick()) ---
    election_deadline_ms: float = 0.0
    next_heartbeat_ms: float = 0.0

    def __post_init__(self) -> None:
        self.election_deadline_ms = self._new_election_deadline(0.0)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def persistent_snapshot(self) -> dict:
        """Everything that MUST be durable before this node acknowledges any
        RPC (paper §5.6: currentTerm, votedFor, log must survive a crash, or
        a restarted node could vote twice in the same term / forget a
        committed entry). The harness/transport is responsible for writing
        this out before releasing this call's returned envelopes."""
        return {
            "node_id": self.node_id,
            "current_term": self.current_term,
            "voted_for": self.voted_for,
            "log_entries": list(self.log.entries),
        }

    @classmethod
    def restore(cls, snapshot: dict, peers: tuple[str, ...], rng: random.Random, **kwargs) -> "RaftNode":
        """Reconstruct a node after a simulated crash from its last durable
        snapshot. Volatile state (commit_index, role, leader knowledge) is
        NOT restored -- per the paper, a restarted server always rejoins as
        a follower with commit_index reset to 0 and re-learns everything
        volatile from the current leader's RPCs, which is what makes crash
        recovery safe without persisting volatile state at all."""
        node = cls(node_id=snapshot["node_id"], peers=peers, rng=rng, **kwargs)
        node.current_term = snapshot["current_term"]
        node.voted_for = snapshot["voted_for"]
        node.log = RaftLog(entries=list(snapshot["log_entries"]))
        return node

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _new_election_deadline(self, now_ms: float) -> float:
        return now_ms + self.rng.uniform(*self.election_timeout_range_ms)

    def _become_follower(self, term: int) -> None:
        stepping_down_leader = self.role == Role.LEADER
        self.role = Role.FOLLOWER
        self.current_term = term
        self.voted_for = None
        self.votes_received = set()
        if stepping_down_leader:
            self.leader_id = None

    def _envelope(self, dst: str, payload) -> Envelope:
        return Envelope(src=self.node_id, dst=dst, payload=payload)

    def _majority(self) -> int:
        return (len(self.peers) + 1) // 2 + 1

    # ------------------------------------------------------------------
    # tick: election timeouts + leader heartbeats
    # ------------------------------------------------------------------

    def tick(self, now_ms: float) -> list[Envelope]:
        out: list[Envelope] = []
        if self.role in (Role.FOLLOWER, Role.CANDIDATE) and now_ms >= self.election_deadline_ms:
            out.extend(self._start_election(now_ms))
        elif self.role == Role.LEADER and now_ms >= self.next_heartbeat_ms:
            out.extend(self._broadcast_append_entries())
            self.next_heartbeat_ms = now_ms + self.heartbeat_interval_ms
        return out

    def _start_election(self, now_ms: float) -> list[Envelope]:
        self.role = Role.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.votes_received = {self.node_id}
        self.leader_id = None
        self.election_deadline_ms = self._new_election_deadline(now_ms)
        args = RequestVoteArgs(
            term=self.current_term,
            candidate_id=self.node_id,
            last_log_index=self.log.last_index,
            last_log_term=self.log.last_term,
        )
        return [self._envelope(peer, args) for peer in self.peers]

    # ------------------------------------------------------------------
    # single dispatch entry point for inbound messages
    # ------------------------------------------------------------------

    def on_message(self, envelope: Envelope, now_ms: float) -> list[Envelope]:
        payload = envelope.payload
        if isinstance(payload, RequestVoteArgs):
            return self._on_request_vote(payload, now_ms)
        if isinstance(payload, RequestVoteReply):
            return self._on_request_vote_reply(payload, now_ms)
        if isinstance(payload, AppendEntriesArgs):
            return self._on_append_entries(payload, now_ms)
        if isinstance(payload, AppendEntriesReply):
            return self._on_append_entries_reply(payload, now_ms)
        raise TypeError(f"unknown payload type {type(payload)!r}")

    # ------------------------------------------------------------------
    # RequestVote
    # ------------------------------------------------------------------

    def _on_request_vote(self, args: RequestVoteArgs, now_ms: float) -> list[Envelope]:
        if args.term > self.current_term:
            self._become_follower(args.term)

        if args.term < self.current_term:
            reply = RequestVoteReply(term=self.current_term, vote_granted=False, voter_id=self.node_id)
            return [self._envelope(args.candidate_id, reply)]

        log_ok = candidate_log_is_up_to_date(args.last_log_term, args.last_log_index, self.log.last_term, self.log.last_index)
        can_vote = self.voted_for in (None, args.candidate_id)
        grant = args.term == self.current_term and can_vote and log_ok

        if grant:
            self.voted_for = args.candidate_id
            # Resetting the election timer here (not just on AppendEntries)
            # is a deliberate choice, not an oversight: without it, a
            # follower that just granted a vote can still time out and
            # start its OWN competing election a moment later, needlessly
            # extending an already-in-progress election round.
            self.election_deadline_ms = self._new_election_deadline(now_ms)

        reply = RequestVoteReply(term=self.current_term, vote_granted=grant, voter_id=self.node_id)
        return [self._envelope(args.candidate_id, reply)]

    def _on_request_vote_reply(self, reply: RequestVoteReply, now_ms: float) -> list[Envelope]:
        if reply.term > self.current_term:
            self._become_follower(reply.term)
            return []
        if self.role != Role.CANDIDATE or reply.term != self.current_term or not reply.vote_granted:
            return []

        self.votes_received.add(reply.voter_id)
        if len(self.votes_received) >= self._majority():
            return self._become_leader(now_ms)
        return []

    def _become_leader(self, now_ms: float) -> list[Envelope]:
        self.role = Role.LEADER
        self.leader_id = self.node_id
        self.next_index = {peer: self.log.last_index + 1 for peer in self.peers}
        self.match_index = {peer: 0 for peer in self.peers}
        # A no-op entry stamped with the new term, committed like any other
        # entry. This gives the leader a same-term entry to replicate
        # immediately, which is what lets it safely advance commit_index
        # past any uncommitted entries inherited from a previous leader
        # (§5.4.2) without waiting for the first real client write.
        self.log.append(LogEntry(term=self.current_term, index=self.log.last_index + 1, command=NOOP))
        self.next_heartbeat_ms = now_ms  # send immediately, not after one interval
        return self._broadcast_append_entries()

    def _broadcast_append_entries(self) -> list[Envelope]:
        out: list[Envelope] = []
        for peer in self.peers:
            out.append(self._append_entries_for(peer))
        return out

    def _append_entries_for(self, peer: str) -> Envelope:
        next_idx = self.next_index.get(peer, self.log.last_index + 1)
        prev_index = next_idx - 1
        prev_term = self.log.term_at(prev_index) or 0
        args = AppendEntriesArgs(
            term=self.current_term,
            leader_id=self.node_id,
            prev_log_index=prev_index,
            prev_log_term=prev_term,
            entries=self.log.slice_from(next_idx),
            leader_commit=self.commit_index,
        )
        return self._envelope(peer, args)

    # ------------------------------------------------------------------
    # AppendEntries
    # ------------------------------------------------------------------

    def _on_append_entries(self, args: AppendEntriesArgs, now_ms: float) -> list[Envelope]:
        if args.term > self.current_term:
            self._become_follower(args.term)
        elif args.term == self.current_term and self.role == Role.CANDIDATE:
            # "If AppendEntries RPC received from new leader: convert to
            # follower" (Figure 2, Candidates) -- applies even when the
            # term is EQUAL, not just greater, because another candidate
            # may have already won this exact term's election.
            self.role = Role.FOLLOWER

        if args.term < self.current_term:
            reply = AppendEntriesReply(term=self.current_term, success=False, follower_id=self.node_id)
            return [self._envelope(args.leader_id, reply)]

        # A valid leader for our term: acknowledge it and reset the
        # election clock -- this is the "leader is alive" signal.
        self.leader_id = args.leader_id
        self.election_deadline_ms = self._new_election_deadline(now_ms)

        prev_term_here = self.log.term_at(args.prev_log_index)
        consistent = prev_term_here is not None and prev_term_here == args.prev_log_term
        if not consistent:
            conflict_term, conflict_index = self.log.conflict_info_for(args.prev_log_index)
            reply = AppendEntriesReply(
                term=self.current_term,
                success=False,
                follower_id=self.node_id,
                conflict_term=conflict_term,
                conflict_index=conflict_index,
            )
            return [self._envelope(args.leader_id, reply)]

        self.log.merge_from_leader(args.prev_log_index, args.entries)

        if args.leader_commit > self.commit_index:
            self.commit_index = min(args.leader_commit, self.log.last_index)

        reply = AppendEntriesReply(
            term=self.current_term,
            success=True,
            follower_id=self.node_id,
            match_index=args.prev_log_index + len(args.entries),
        )
        return [self._envelope(args.leader_id, reply)]

    def _on_append_entries_reply(self, reply: AppendEntriesReply, now_ms: float) -> list[Envelope]:
        if reply.term > self.current_term:
            self._become_follower(reply.term)
            return []
        if self.role != Role.LEADER or reply.term != self.current_term:
            return []

        peer = reply.follower_id
        if reply.success:
            assert reply.match_index is not None
            self.match_index[peer] = max(self.match_index.get(peer, 0), reply.match_index)
            self.next_index[peer] = self.match_index[peer] + 1
            self._advance_commit_index()
            return []

        # Fast backup (§5.3 / student guide): jump nextIndex using the
        # follower's conflict hint instead of decrementing by one and
        # re-probing every entry, which is O(log length) round trips on a
        # long divergent log.
        if reply.conflict_term is not None:
            last_index_of_conflict_term = None
            for idx in range(self.log.last_index, 0, -1):
                if self.log.term_at(idx) == reply.conflict_term:
                    last_index_of_conflict_term = idx
                    break
            if last_index_of_conflict_term is not None:
                self.next_index[peer] = last_index_of_conflict_term + 1
            else:
                self.next_index[peer] = reply.conflict_index or 1
        else:
            self.next_index[peer] = max(1, reply.conflict_index or (self.next_index.get(peer, 1) - 1))

        return [self._append_entries_for(peer)]

    def _advance_commit_index(self) -> None:
        # §5.3/§5.4.2: only ever commit by counting replicas for an entry
        # from the LEADER'S CURRENT TERM. Committing an older-term entry
        # this way is the classic Raft correctness bug (paper Figure 8) --
        # a majority can replicate an old-term entry and then have it
        # overwritten by a later leader that was unaware it was "committed."
        for candidate_index in range(self.log.last_index, self.commit_index, -1):
            if self.log.term_at(candidate_index) != self.current_term:
                continue
            replicated_count = 1 + sum(1 for peer in self.peers if self.match_index.get(peer, 0) >= candidate_index)
            if replicated_count >= self._majority():
                self.commit_index = candidate_index
                break

    # ------------------------------------------------------------------
    # client-facing
    # ------------------------------------------------------------------

    def propose(self, command: object, now_ms: float) -> ProposeResult:
        if self.role != Role.LEADER:
            return ProposeResult(accepted=False, index=None, term=None, leader_hint=self.leader_id, envelopes=[])
        entry = LogEntry(term=self.current_term, index=self.log.last_index + 1, command=command)
        self.log.append(entry)
        envelopes = self._broadcast_append_entries()
        return ProposeResult(accepted=True, index=entry.index, term=entry.term, leader_hint=self.node_id, envelopes=envelopes)

    def take_newly_committed(self) -> list[LogEntry]:
        """Entries that became committed since the last call, in order.
        Advances last_applied. The KV state-machine layer (not core Raft)
        owns turning these into get/put/delete effects -- keeping that
        separation is what let features.py-style formula testing apply
        here too: this method is tested purely on commit_index bookkeeping,
        independent of what a "command" even means."""
        if self.commit_index <= self.last_applied:
            return []
        newly = [self.log.entry_at(i) for i in range(self.last_applied + 1, self.commit_index + 1)]
        self.last_applied = self.commit_index
        return [e for e in newly if e is not None]
