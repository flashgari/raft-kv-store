"""A deterministic, in-process simulated network for driving a cluster of
`RaftNode`s through fault injection (drop, delay, duplicate, partition,
crash/restart) without a single real socket or `sleep()` call.

This is what makes it possible to run thousands of randomized chaos trials
in well under a second (see tests/test_chaos.py) and get bit-for-bit
reproducible failures from a random seed -- a real-network integration
test suite could exercise the same scenarios only a handful of times in
the same wall-clock budget, and a bug that only manifests 1-in-2000 trials
would be extremely expensive to catch that way.
"""

from __future__ import annotations

import heapq
import itertools
import random
from dataclasses import dataclass, field

from typing import Callable

from .log import RaftLog
from .messages import Envelope, LogEntry
from .node import RaftNode


@dataclass
class NetworkConditions:
    latency_ms_range: tuple[float, float] = (1.0, 10.0)
    drop_probability: float = 0.0
    duplicate_probability: float = 0.0


@dataclass
class SimulatedCluster:
    node_ids: tuple[str, ...]
    rng: random.Random
    election_timeout_range_ms: tuple[float, float] = (150.0, 300.0)
    heartbeat_interval_ms: float = 50.0
    conditions: NetworkConditions = field(default_factory=NetworkConditions)

    now_ms: float = 0.0
    nodes: dict[str, RaftNode] = field(default_factory=dict)
    # node_id -> "durable storage" snapshot, written every time that node's
    # state changes -- read back on restart() to model crash recovery.
    stable_storage: dict[str, dict] = field(default_factory=dict)
    crashed: set[str] = field(default_factory=set)
    # node_id -> partition group label; nodes only exchange messages with
    # peers sharing the same label. All-same-label == fully connected.
    partition_of: dict[str, str] = field(default_factory=dict)

    _inbox: list[tuple[float, int, Envelope]] = field(default_factory=list)
    _tie_breaker: itertools.count = field(default_factory=itertools.count)
    committed_log_by_node: dict[str, list] = field(default_factory=dict)
    # Records (term, leader_id) the FIRST time each term acquires a leader,
    # for the Election Safety check (at most one leader per term, ever).
    leader_history: dict[int, str] = field(default_factory=dict)
    # Optional hook: called as apply_callback(node_id, entry) for every
    # newly-committed entry on every node, in commit order -- this is how
    # a layer above Raft (e.g. kvstore.KVStateMachine) gets to apply
    # entries exactly once each, without a second, separately-draining
    # call to take_newly_committed() racing the safety-check bookkeeping
    # below that already drains it once per node.
    apply_callback: Callable[[str, LogEntry], None] | None = None

    def __post_init__(self) -> None:
        for nid in self.node_ids:
            peers = tuple(p for p in self.node_ids if p != nid)
            node = RaftNode(
                node_id=nid,
                peers=peers,
                rng=random.Random(self.rng.random()),
                election_timeout_range_ms=self.election_timeout_range_ms,
                heartbeat_interval_ms=self.heartbeat_interval_ms,
            )
            self.nodes[nid] = node
            self.stable_storage[nid] = node.persistent_snapshot()
            self.partition_of[nid] = "*"
            self.committed_log_by_node[nid] = []

    # ------------------------------------------------------------------
    # fault injection controls
    # ------------------------------------------------------------------

    def crash(self, node_id: str) -> None:
        """Stop delivering to/scheduling from this node. Its last durable
        snapshot (already in stable_storage) is preserved; all volatile
        state (commit_index, role, in-flight timers) is discarded, exactly
        as a real process crash would."""
        self.crashed.add(node_id)

    def restart(self, node_id: str) -> None:
        snapshot = self.stable_storage[node_id]
        peers = tuple(p for p in self.node_ids if p != node_id)
        self.nodes[node_id] = RaftNode.restore(
            snapshot,
            peers,
            rng=random.Random(self.rng.random()),
            election_timeout_range_ms=self.election_timeout_range_ms,
            heartbeat_interval_ms=self.heartbeat_interval_ms,
        )
        self.crashed.discard(node_id)

    def partition(self, groups: list[set[str]]) -> None:
        for i, group in enumerate(groups):
            for nid in group:
                self.partition_of[nid] = f"group{i}"

    def heal_partition(self) -> None:
        for nid in self.node_ids:
            self.partition_of[nid] = "*"

    # ------------------------------------------------------------------
    # message scheduling
    # ------------------------------------------------------------------

    def _schedule(self, envelope: Envelope) -> None:
        if self.partition_of[envelope.src] != self.partition_of[envelope.dst]:
            return  # partitioned apart; dropped
        if self.rng.random() < self.conditions.drop_probability:
            return
        copies = 2 if self.rng.random() < self.conditions.duplicate_probability else 1
        for _ in range(copies):
            latency = self.rng.uniform(*self.conditions.latency_ms_range)
            deliver_at = self.now_ms + latency
            heapq.heappush(self._inbox, (deliver_at, next(self._tie_breaker), envelope))

    def _persist(self, node_id: str) -> None:
        self.stable_storage[node_id] = self.nodes[node_id].persistent_snapshot()

    # ------------------------------------------------------------------
    # driving the simulation
    # ------------------------------------------------------------------

    def step(self, dt_ms: float) -> None:
        self.now_ms += dt_ms

        while self._inbox and self._inbox[0][0] <= self.now_ms:
            _, _, envelope = heapq.heappop(self._inbox)
            if envelope.dst in self.crashed or envelope.dst not in self.nodes:
                continue
            node = self.nodes[envelope.dst]
            out = node.on_message(envelope, self.now_ms)
            self._persist(envelope.dst)  # durable state before releasing replies (§5.6)
            for e in out:
                self._schedule(e)
            self._record_leader_and_commits(envelope.dst, node)

        for nid, node in list(self.nodes.items()):
            if nid in self.crashed:
                continue
            out = node.tick(self.now_ms)
            if out:
                self._persist(nid)
                for e in out:
                    self._schedule(e)
            self._record_leader_and_commits(nid, node)

    def run_for(self, duration_ms: float, dt_ms: float = 5.0) -> None:
        steps = int(duration_ms // dt_ms)
        for _ in range(steps):
            self.step(dt_ms)

    def _record_leader_and_commits(self, node_id: str, node: RaftNode) -> None:
        from .node import Role

        if node.role == Role.LEADER:
            recorded = self.leader_history.get(node.current_term)
            if recorded is not None and recorded != node_id:
                raise AssertionError(
                    f"ELECTION SAFETY VIOLATED: term {node.current_term} already had leader "
                    f"{recorded!r}, but {node_id!r} also believes it is leader"
                )
            self.leader_history[node.current_term] = node_id

        for entry in node.take_newly_committed():
            log = self.committed_log_by_node[node_id]
            if len(log) < entry.index:
                log.extend([None] * (entry.index - len(log)))
            if log[entry.index - 1] is not None and log[entry.index - 1] != entry:
                raise AssertionError(
                    f"STATE MACHINE SAFETY VIOLATED: {node_id!r} applied a different command "
                    f"at index {entry.index} than it did before"
                )
            log[entry.index - 1] = entry
            if self.apply_callback is not None:
                self.apply_callback(node_id, entry)

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def current_leader(self) -> str | None:
        from .node import Role

        leaders = [nid for nid, n in self.nodes.items() if nid not in self.crashed and n.role == Role.LEADER]
        # Note: during a leadership change there can briefly be zero
        # leaders (never more than one, checked continuously above).
        return leaders[0] if leaders else None

    def propose_via_leader(self, command: object):
        leader_id = self.current_leader()
        if leader_id is None:
            return None
        result = self.nodes[leader_id].propose(command, self.now_ms)
        if result.accepted:
            self._persist(leader_id)
            for e in result.envelopes:
                self._schedule(e)
        return result

    def check_log_matching_property(self) -> None:
        """Cross-node invariant, checked at the end of a scenario (cheap to
        run continuously, but clearer as an explicit assertion at
        checkpoints): if two logs both have an entry at the same index with
        the same term, every entry up to and including that index must be
        identical across both logs (paper §5.3)."""
        alive = [n for nid, n in self.nodes.items() if nid not in self.crashed]
        for a, b in itertools.combinations(alive, 2):
            upto = min(a.log.last_index, b.log.last_index)
            for idx in range(1, upto + 1):
                ea, eb = a.log.entry_at(idx), b.log.entry_at(idx)
                if ea is None or eb is None:
                    continue
                if ea.term == eb.term:
                    assert ea.command == eb.command, (
                        f"LOG MATCHING VIOLATED at index {idx}: {a.node_id}={ea!r} vs {b.node_id}={eb!r}"
                    )
                    # And once terms match at idx, everything before must match too.
                    for j in range(1, idx):
                        ea2, eb2 = a.log.entry_at(j), b.log.entry_at(j)
                        assert ea2 == eb2, f"LOG MATCHING VIOLATED at prefix index {j} given match at {idx}"
