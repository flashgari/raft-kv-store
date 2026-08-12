"""The real asyncio TCP transport: wires a plain `RaftNode` up to actual
sockets for the live multi-process demo (scripts/run_node.py). Everything
node-decision-making stays inside `RaftNode` -- this module's entire job
is I/O: accept peer connections, dial out to peers, run the tick loop
against the wall clock, and serve a small client protocol on a second
port. The chaos/linearizability test suite exercises the algorithm far
more thoroughly than a live socket demo ever could (thousands of
randomized trials vs. a handful of manual runs), so this layer is
deliberately thin -- it is not where this project's correctness argument
lives.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path

from .kvstore import ClientCommand, KVStateMachine
from .messages import Envelope, LogEntry
from .node import NOOP, RaftNode
from .wire import decode_client_request, decode_envelope, encode_client_reply, encode_envelope, read_framed


def _now_ms() -> float:
    return time.monotonic() * 1000.0


class DurableStorage:
    """Persists (current_term, voted_for, log) to a local JSON file after
    every state-mutating call, and reloads it on process start -- the real
    counterpart to SimulatedCluster.stable_storage in tests. A single
    `os.replace`-based atomic write avoids leaving a half-written file if
    the process is killed mid-write."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        raw = json.loads(self.path.read_text())
        entries = [LogEntry(term=e["term"], index=e["index"], command=self._decode_command(e["command"])) for e in raw["log_entries"]]
        return {"node_id": raw["node_id"], "current_term": raw["current_term"], "voted_for": raw["voted_for"], "log_entries": entries}

    def save(self, snapshot: dict) -> None:
        encoded = {
            "node_id": snapshot["node_id"],
            "current_term": snapshot["current_term"],
            "voted_for": snapshot["voted_for"],
            "log_entries": [self._encode_entry(e) for e in snapshot["log_entries"]],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(encoded))
        tmp.replace(self.path)

    @staticmethod
    def _encode_entry(entry: LogEntry) -> dict:
        cmd = entry.command
        if cmd is NOOP:
            encoded_cmd = "__noop__"
        elif cmd is None:
            encoded_cmd = None  # the log sentinel at index 0
        else:
            encoded_cmd = {"client_id": cmd.client_id, "seq": cmd.seq, "op": cmd.op, "key": cmd.key, "value": cmd.value}
        return {"term": entry.term, "index": entry.index, "command": encoded_cmd}

    @staticmethod
    def _decode_command(raw) -> object:
        if raw is None:
            return None
        if raw == "__noop__":
            return NOOP
        return ClientCommand(client_id=raw["client_id"], seq=raw["seq"], op=raw["op"], key=raw["key"], value=raw["value"])


class AsyncRaftServer:
    def __init__(
        self,
        node_id: str,
        peer_addrs: dict[str, tuple[str, int]],
        listen_port: int,
        client_port: int,
        data_dir: Path,
        election_timeout_range_ms: tuple[float, float] = (300.0, 600.0),
        heartbeat_interval_ms: float = 100.0,
        tick_interval_ms: float = 20.0,
    ):
        self.node_id = node_id
        self.peer_addrs = peer_addrs
        self.listen_port = listen_port
        self.client_port = client_port
        self.tick_interval_ms = tick_interval_ms
        self.storage = DurableStorage(data_dir / f"{node_id}.json")

        peers = tuple(peer_addrs.keys())
        snapshot = self.storage.load()
        if snapshot is not None:
            self.node = RaftNode.restore(
                snapshot,
                peers,
                rng=random.Random(),
                election_timeout_range_ms=election_timeout_range_ms,
                heartbeat_interval_ms=heartbeat_interval_ms,
            )
            print(f"[{node_id}] restored from {self.storage.path}: term={self.node.current_term}, log_len={len(self.node.log)}")
        else:
            self.node = RaftNode(
                node_id=node_id,
                peers=peers,
                rng=random.Random(),
                election_timeout_range_ms=election_timeout_range_ms,
                heartbeat_interval_ms=heartbeat_interval_ms,
            )

        self.state_machine = KVStateMachine()
        self._writers: dict[str, asyncio.StreamWriter] = {}
        self._pending_replies: dict[tuple[str, int], object] = {}  # (client_id, seq) -> Reply, for local wakeups

    # ------------------------------------------------------------------

    def _persist(self) -> None:
        self.storage.save(self.node.persistent_snapshot())

    async def start(self) -> None:
        peer_server = await asyncio.start_server(self._handle_peer_connection, host="127.0.0.1", port=self.listen_port)
        client_server = await asyncio.start_server(self._handle_client_connection, host="127.0.0.1", port=self.client_port)
        print(f"[{self.node_id}] listening: peers=127.0.0.1:{self.listen_port} clients=127.0.0.1:{self.client_port}")
        asyncio.create_task(self._tick_loop())
        async with peer_server, client_server:
            await asyncio.gather(peer_server.serve_forever(), client_server.serve_forever())

    # ------------------------------------------------------------------
    # peer <-> peer
    # ------------------------------------------------------------------

    async def _handle_peer_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                body = await read_framed(reader)
                envelope = decode_envelope(body)
                self._deliver(envelope)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()

    def _deliver(self, envelope: Envelope) -> None:
        out = self.node.on_message(envelope, _now_ms())
        self._persist()
        self._apply_newly_committed()
        for e in out:
            asyncio.create_task(self._send(e))

    async def _get_writer(self, peer: str) -> asyncio.StreamWriter | None:
        writer = self._writers.get(peer)
        if writer is not None and not writer.is_closing():
            return writer
        host, port = self.peer_addrs[peer]
        try:
            _, writer = await asyncio.open_connection(host, port)
        except OSError:
            return None
        self._writers[peer] = writer
        return writer

    async def _send(self, envelope: Envelope) -> None:
        writer = await self._get_writer(envelope.dst)
        if writer is None:
            return  # peer unreachable; Raft is designed to tolerate lost messages
        try:
            writer.write(encode_envelope(envelope))
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            self._writers.pop(envelope.dst, None)

    # ------------------------------------------------------------------
    # tick loop
    # ------------------------------------------------------------------

    async def _tick_loop(self) -> None:
        while True:
            await asyncio.sleep(self.tick_interval_ms / 1000.0)
            out = self.node.tick(_now_ms())
            if out:
                self._persist()
                for e in out:
                    asyncio.create_task(self._send(e))
            self._apply_newly_committed()

    def _apply_newly_committed(self) -> None:
        for entry in self.node.take_newly_committed():
            self.state_machine.apply(entry)

    # ------------------------------------------------------------------
    # client <-> cluster
    # ------------------------------------------------------------------

    async def _handle_client_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                body = await read_framed(reader)
                command = decode_client_request(body)
                reply_bytes = await self._handle_client_command(command)
                writer.write(reply_bytes)
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()

    async def _handle_client_command(self, command: ClientCommand, max_wait_s: float = 5.0) -> bytes:
        from .node import Role

        if self.node.role != Role.LEADER:
            return encode_client_reply(ok=False, value=None, existed=False, leader_hint=self.node.leader_id, error="not_leader")

        result = self.node.propose(command, _now_ms())
        if not result.accepted:
            return encode_client_reply(ok=False, value=None, existed=False, leader_hint=result.leader_hint, error="not_leader")
        self._persist()
        for envelope in result.envelopes:
            asyncio.create_task(self._send(envelope))

        deadline = time.monotonic() + max_wait_s
        while time.monotonic() < deadline:
            await asyncio.sleep(0.01)
            if self.node.role != Role.LEADER:
                return encode_client_reply(ok=False, value=None, existed=False, leader_hint=self.node.leader_id, error="leadership_changed")
            if self.state_machine.last_seq_by_client.get(command.client_id, 0) >= command.seq:
                reply = self.state_machine.last_reply_by_client[command.client_id]
                return encode_client_reply(ok=reply.ok, value=reply.value, existed=reply.existed, leader_hint=self.node.node_id)
        return encode_client_reply(ok=False, value=None, existed=False, leader_hint=self.node.node_id, error="timeout")
