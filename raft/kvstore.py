"""A linearizable key-value store layered on the replicated log.

Two design choices worth calling out because they're the actual hard part
of building a *correct* replicated KV store on top of a consensus log --
Raft only guarantees that every node applies the same commands in the
same order, not that a naive client protocol built on top of it is
linearizable:

1. **Reads go through the log too.** A `Get` is implemented as a log entry
   just like `Put`/`Delete` (it just doesn't mutate `store`). This is the
   simplest possible way to make reads linearizable: since it is ordered
   in the same commit stream as every write, a `Get` that commits after a
   `Put` is guaranteed to observe it, with no separate leader-lease or
   read-index protocol to get right. The cost is latency (a read pays a
   full replication round trip instead of being served from local leader
   state) -- documented as a deliberate simplicity-over-latency tradeoff
   in the README, not left unstated.
2. **Client commands are deduplicated by (client_id, sequence).** A client
   whose leader crashes mid-request cannot tell whether its command
   committed or not, and the only safe thing to do is retry. Without
   dedup, a retried `Put` could apply twice. `KVStateMachine` remembers
   the last sequence number applied per client and, on a duplicate,
   returns the ORIGINAL cached reply instead of re-applying -- this is
   the same at-most-once scheme used in MIT 6.824's KV service labs and
   in real production systems (e.g. etcd's per-lease request dedup).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .messages import LogEntry
from .node import NOOP
from .simulate import SimulatedCluster


@dataclass(frozen=True)
class ClientCommand:
    client_id: str
    seq: int
    op: str  # "put" | "delete" | "get"
    key: str
    value: str | None = None


@dataclass(frozen=True)
class Reply:
    ok: bool
    value: str | None = None
    existed: bool = True


@dataclass
class KVStateMachine:
    store: dict[str, str] = field(default_factory=dict)
    last_seq_by_client: dict[str, int] = field(default_factory=dict)
    last_reply_by_client: dict[str, Reply] = field(default_factory=dict)
    applied_count: int = 0

    def apply(self, entry: LogEntry) -> Reply | None:
        command = entry.command
        if command is NOOP or not isinstance(command, ClientCommand):
            return None  # leader no-op entries never reach client state

        last_seq = self.last_seq_by_client.get(command.client_id, 0)
        if command.seq <= last_seq:
            # Duplicate/retried command already applied -- return the
            # cached reply instead of re-applying (see module docstring).
            # Deliberately does NOT bump applied_count: that counter is
            # used by tests/verification to prove dedup is actually
            # skipping work, not just returning the same answer twice by
            # coincidence (an earlier version incremented it before this
            # check and always reported the command as newly applied --
            # see README, "Bugs found and fixed").
            return self.last_reply_by_client.get(command.client_id)

        self.applied_count += 1
        if command.op == "put":
            self.store[command.key] = command.value
            reply = Reply(ok=True, value=command.value)
        elif command.op == "delete":
            existed = command.key in self.store
            self.store.pop(command.key, None)
            reply = Reply(ok=True, existed=existed)
        elif command.op == "get":
            reply = Reply(ok=command.key in self.store, value=self.store.get(command.key))
        else:
            raise ValueError(f"unknown op {command.op!r}")

        self.last_seq_by_client[command.client_id] = command.seq
        self.last_reply_by_client[command.client_id] = reply
        return reply


class KVCluster:
    """Wires a SimulatedCluster's commit stream to one KVStateMachine per
    node, so every node's store evolves identically and independently --
    exactly what lets tests/test_linearizability.py validate the system
    from the outside, as if it were a single register, regardless of
    which physical node answered a given request."""

    def __init__(self, cluster: SimulatedCluster):
        self.cluster = cluster
        self.state_machines: dict[str, KVStateMachine] = {nid: KVStateMachine() for nid in cluster.node_ids}
        cluster.apply_callback = self._on_apply

    def _on_apply(self, node_id: str, entry: LogEntry) -> None:
        self.state_machines[node_id].apply(entry)


class RequestTimeout(Exception):
    pass


class SimulatedKVClient:
    """A client driving a KVCluster over simulated time: proposes through
    whatever node it currently believes is the leader, follows leader-hint
    redirects on rejection, and retries the SAME (client_id, seq) pair on
    timeout so a leader crash mid-request is always safe to retry."""

    def __init__(self, kv_cluster: KVCluster, client_id: str, step_ms: float = 10.0):
        self.kv_cluster = kv_cluster
        self.cluster = kv_cluster.cluster
        self.client_id = client_id
        self.step_ms = step_ms
        self._seq = 0
        self._known_leader: str | None = None

    def _request_iter(self, op: str, key: str, value: str | None, max_wait_ms: float):
        """Generator form of a request: yields once per simulated step
        instead of calling `cluster.step()` itself, so a single caller can
        own the shared clock and interleave many clients' generators in
        the same round -- this is what makes it possible to produce a
        history with genuinely OVERLAPPING operations (client B's call
        starts before client A's returns) without real threads, which is
        the only kind of history a linearizability checker is interesting
        on. `_request` below is a thin blocking wrapper around this for
        single-client callers (tests/test_kvstore.py) that don't care
        about concurrency."""
        self._seq += 1
        seq = self._seq
        command = ClientCommand(client_id=self.client_id, seq=seq, op=op, key=key, value=value)
        waited = 0.0
        while waited < max_wait_ms:
            leader = self._known_leader or self.cluster.current_leader()
            if leader is None or leader not in self.cluster.nodes or leader in self.cluster.crashed:
                yield
                waited += self.step_ms
                self._known_leader = None
                continue

            result = self.cluster.nodes[leader].propose(command, self.cluster.now_ms)
            if not result.accepted:
                self._known_leader = result.leader_hint
                yield
                waited += self.step_ms
                continue

            self.cluster._persist(leader)
            for envelope in result.envelopes:
                self.cluster._schedule(envelope)

            # Wait for OUR OWN command to actually be applied -- checking
            # last_seq_by_client rather than "index committed" catches the
            # case where a later client's command commits first and we'd
            # otherwise return before ours has actually taken effect.
            while waited < max_wait_ms:
                yield
                waited += self.step_ms
                sm = self.kv_cluster.state_machines[leader]
                if sm.last_seq_by_client.get(self.client_id, 0) >= seq:
                    return sm.last_reply_by_client[self.client_id]
                from .node import Role

                if self.cluster.nodes[leader].role != Role.LEADER:
                    self._known_leader = None
                    break  # leadership moved mid-flight; re-propose against the new leader
        raise RequestTimeout(f"{self.client_id} request {op} {key!r} did not complete within {max_wait_ms} ms")

    def _request(self, op: str, key: str, value: str | None, max_wait_ms: float) -> Reply:
        gen = self._request_iter(op, key, value, max_wait_ms)
        try:
            while True:
                next(gen)
                self.cluster.step(self.step_ms)
        except StopIteration as stop:
            return stop.value

    def put(self, key: str, value: str, max_wait_ms: float = 5000.0) -> Reply:
        return self._request("put", key, value, max_wait_ms)

    def delete(self, key: str, max_wait_ms: float = 5000.0) -> Reply:
        return self._request("delete", key, None, max_wait_ms)

    def get(self, key: str, max_wait_ms: float = 5000.0) -> Reply:
        return self._request("get", key, None, max_wait_ms)


@dataclass(frozen=True)
class Operation:
    """One completed client operation, timestamped in the cluster's
    simulated-time coordinates -- the unit the linearizability checker
    reasons about."""

    client_id: str
    op: str
    key: str
    value: str | None
    start_ms: float
    end_ms: float
    reply: Reply


class ConcurrentWorkloadDriver:
    """Runs several clients' operation sequences AGAINST ONE SHARED
    simulated clock, interleaved round-by-round, so their operations can
    genuinely overlap in simulated time -- e.g. client B's PUT starts
    before client A's GET (issued earlier) has returned. Each client's own
    operations stay in program order (a client never has two requests in
    flight at once, matching how a real client library is used), but
    different clients' requests race freely, which is exactly the
    scenario a linearizability checker needs to be a meaningful test
    rather than a no-op on an already-sequential history.
    """

    def __init__(self, kv_cluster: KVCluster, step_ms: float = 10.0):
        self.kv_cluster = kv_cluster
        self.cluster = kv_cluster.cluster
        self.step_ms = step_ms
        # Cached across `run()` calls, keyed by client_id -- NOT recreated
        # per call. A fresh SimulatedKVClient starts its sequence counter
        # at 0, so calling run() twice for the same client_id with a
        # fresh client would reissue seq=1 for a genuinely new command,
        # colliding with KVStateMachine's own (client_id, seq) dedup and
        # causing the second command to be silently treated as a replay
        # of the first (see README, "Bugs found and fixed" -- caught by
        # tests/test_linearizability.py::test_kv_cluster_is_linearizable_across_a_leader_crash_mid_workload,
        # which runs one workload across two run() calls split by a
        # leader crash).
        self._clients: dict[str, SimulatedKVClient] = {}

    def run(self, workloads: dict[str, list[tuple[str, str, str | None]]], max_wait_ms: float = 20000.0) -> list[Operation]:
        """`workloads`: client_id -> ordered list of (op, key, value) to
        issue one after another. Returns the full observed history."""
        history: list[Operation] = []
        remaining = {cid: list(ops) for cid, ops in workloads.items()}
        for cid in workloads:
            if cid not in self._clients:
                self._clients[cid] = SimulatedKVClient(self.kv_cluster, cid, step_ms=self.step_ms)
        clients = self._clients
        active: dict[str, dict] = {}

        def start_next(cid: str) -> dict | None:
            if not remaining[cid]:
                return None
            op, key, value = remaining[cid].pop(0)
            gen = clients[cid]._request_iter(op, key, value, max_wait_ms)
            return {"gen": gen, "op": op, "key": key, "value": value, "start": None}

        for cid in workloads:
            active[cid] = start_next(cid)

        total_ops = sum(len(v) for v in workloads.values())
        completed = 0
        rounds = 0
        max_rounds = int(max_wait_ms / self.step_ms) * max(1, total_ops)
        while any(v is not None for v in active.values()) and rounds < max_rounds:
            self.cluster.step(self.step_ms)
            rounds += 1
            for cid, info in list(active.items()):
                if info is None:
                    continue
                if info["start"] is None:
                    info["start"] = self.cluster.now_ms
                try:
                    next(info["gen"])
                except StopIteration as stop:
                    history.append(
                        Operation(
                            client_id=cid,
                            op=info["op"],
                            key=info["key"],
                            value=info["value"],
                            start_ms=info["start"],
                            end_ms=self.cluster.now_ms,
                            reply=stop.value,
                        )
                    )
                    completed += 1
                    active[cid] = start_next(cid)
        if any(v is not None for v in active.values()):
            raise RequestTimeout(f"workload did not complete within {max_rounds} rounds ({completed}/{total_ops} ops finished)")
        return history
