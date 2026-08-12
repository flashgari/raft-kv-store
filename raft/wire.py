"""JSON wire encoding for inter-node envelopes and the client<->cluster
protocol. JSON, not a binary format, is a deliberate choice for this
demo: it keeps every message inspectable with `nc`/Wireshark's text view
without a schema compiler, at the cost of throughput a production system
wouldn't accept -- documented in the README as a scoping decision, not an
oversight. Every message is length-prefixed (4-byte big-endian length
header + JSON body) over a plain TCP stream, so a reader never has to
guess where one message ends and the next begins.
"""

from __future__ import annotations

import json
import struct

from .kvstore import ClientCommand
from .messages import (
    AppendEntriesArgs,
    AppendEntriesReply,
    Envelope,
    LogEntry,
    RequestVoteArgs,
    RequestVoteReply,
)
from .node import NOOP

_NOOP_TAG = "__noop__"


def _encode_command(command: object) -> dict | str:
    if command is NOOP:
        return _NOOP_TAG
    if isinstance(command, ClientCommand):
        return {
            "client_id": command.client_id,
            "seq": command.seq,
            "op": command.op,
            "key": command.key,
            "value": command.value,
        }
    raise TypeError(f"cannot encode command of type {type(command)!r}")


def _decode_command(raw: dict | str) -> object:
    if raw == _NOOP_TAG:
        return NOOP
    return ClientCommand(client_id=raw["client_id"], seq=raw["seq"], op=raw["op"], key=raw["key"], value=raw["value"])


def _encode_entry(entry: LogEntry) -> dict:
    return {"term": entry.term, "index": entry.index, "command": _encode_command(entry.command)}


def _decode_entry(raw: dict) -> LogEntry:
    return LogEntry(term=raw["term"], index=raw["index"], command=_decode_command(raw["command"]))


def encode_envelope(envelope: Envelope) -> bytes:
    payload = envelope.payload
    if isinstance(payload, RequestVoteArgs):
        body = {
            "type": "request_vote_args",
            "term": payload.term,
            "candidate_id": payload.candidate_id,
            "last_log_index": payload.last_log_index,
            "last_log_term": payload.last_log_term,
        }
    elif isinstance(payload, RequestVoteReply):
        body = {"type": "request_vote_reply", "term": payload.term, "vote_granted": payload.vote_granted, "voter_id": payload.voter_id}
    elif isinstance(payload, AppendEntriesArgs):
        body = {
            "type": "append_entries_args",
            "term": payload.term,
            "leader_id": payload.leader_id,
            "prev_log_index": payload.prev_log_index,
            "prev_log_term": payload.prev_log_term,
            "entries": [_encode_entry(e) for e in payload.entries],
            "leader_commit": payload.leader_commit,
        }
    elif isinstance(payload, AppendEntriesReply):
        body = {
            "type": "append_entries_reply",
            "term": payload.term,
            "success": payload.success,
            "follower_id": payload.follower_id,
            "conflict_term": payload.conflict_term,
            "conflict_index": payload.conflict_index,
            "match_index": payload.match_index,
        }
    else:
        raise TypeError(f"cannot encode payload of type {type(payload)!r}")
    envelope_dict = {"src": envelope.src, "dst": envelope.dst, "payload": body}
    return _frame(json.dumps(envelope_dict).encode("utf-8"))


def decode_envelope(data: bytes) -> Envelope:
    raw = json.loads(data.decode("utf-8"))
    body = raw["payload"]
    kind = body["type"]
    if kind == "request_vote_args":
        payload = RequestVoteArgs(body["term"], body["candidate_id"], body["last_log_index"], body["last_log_term"])
    elif kind == "request_vote_reply":
        payload = RequestVoteReply(body["term"], body["vote_granted"], body["voter_id"])
    elif kind == "append_entries_args":
        payload = AppendEntriesArgs(
            term=body["term"],
            leader_id=body["leader_id"],
            prev_log_index=body["prev_log_index"],
            prev_log_term=body["prev_log_term"],
            entries=tuple(_decode_entry(e) for e in body["entries"]),
            leader_commit=body["leader_commit"],
        )
    elif kind == "append_entries_reply":
        payload = AppendEntriesReply(
            term=body["term"],
            success=body["success"],
            follower_id=body["follower_id"],
            conflict_term=body.get("conflict_term"),
            conflict_index=body.get("conflict_index"),
            match_index=body.get("match_index"),
        )
    else:
        raise ValueError(f"unknown envelope payload type {kind!r}")
    return Envelope(src=raw["src"], dst=raw["dst"], payload=payload)


def encode_client_request(client_id: str, seq: int, op: str, key: str, value: str | None) -> bytes:
    return _frame(json.dumps({"client_id": client_id, "seq": seq, "op": op, "key": key, "value": value}).encode("utf-8"))


def decode_client_request(data: bytes) -> ClientCommand:
    raw = json.loads(data.decode("utf-8"))
    return ClientCommand(client_id=raw["client_id"], seq=raw["seq"], op=raw["op"], key=raw["key"], value=raw["value"])


def encode_client_reply(ok: bool, value: str | None, existed: bool, leader_hint: str | None, error: str | None = None) -> bytes:
    return _frame(json.dumps({"ok": ok, "value": value, "existed": existed, "leader_hint": leader_hint, "error": error}).encode("utf-8"))


def decode_client_reply(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"))


def _frame(body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + body


async def read_framed(reader) -> bytes | None:
    header = await reader.readexactly(4)
    (length,) = struct.unpack(">I", header)
    return await reader.readexactly(length)
