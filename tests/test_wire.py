"""Round-trip tests for the JSON wire encoding -- catches serialization
bugs (e.g. the NOOP sentinel, optional fields on AppendEntriesReply) with
plain function calls, no sockets needed."""

from __future__ import annotations

from raft.kvstore import ClientCommand
from raft.messages import AppendEntriesArgs, AppendEntriesReply, Envelope, LogEntry, RequestVoteArgs, RequestVoteReply
from raft.node import NOOP
from raft.wire import decode_client_reply, decode_client_request, decode_envelope, encode_client_reply, encode_client_request, encode_envelope


def _roundtrip(envelope: Envelope) -> Envelope:
    framed = encode_envelope(envelope)
    length = int.from_bytes(framed[:4], "big")
    assert length == len(framed) - 4
    return decode_envelope(framed[4:])


def test_request_vote_args_roundtrips():
    env = Envelope("n1", "n2", RequestVoteArgs(term=5, candidate_id="n1", last_log_index=3, last_log_term=2))
    assert _roundtrip(env) == env


def test_request_vote_reply_roundtrips():
    env = Envelope("n2", "n1", RequestVoteReply(term=5, vote_granted=True, voter_id="n2"))
    assert _roundtrip(env) == env


def test_append_entries_args_with_client_command_entries_roundtrips():
    cmd = ClientCommand(client_id="c1", seq=3, op="put", key="x", value="42")
    entry = LogEntry(term=2, index=7, command=cmd)
    args = AppendEntriesArgs(term=2, leader_id="n1", prev_log_index=6, prev_log_term=2, entries=(entry,), leader_commit=5)
    env = Envelope("n1", "n2", args)
    assert _roundtrip(env) == env


def test_append_entries_args_with_noop_entry_roundtrips():
    entry = LogEntry(term=2, index=1, command=NOOP)
    args = AppendEntriesArgs(term=2, leader_id="n1", prev_log_index=0, prev_log_term=0, entries=(entry,), leader_commit=0)
    env = Envelope("n1", "n2", args)
    got = _roundtrip(env)
    assert got.payload.entries[0].command is NOOP


def test_append_entries_reply_with_optional_fields_none_roundtrips():
    reply = AppendEntriesReply(term=1, success=True, follower_id="n2", match_index=5)
    env = Envelope("n2", "n1", reply)
    assert _roundtrip(env) == env


def test_append_entries_reply_conflict_fields_roundtrip():
    reply = AppendEntriesReply(term=1, success=False, follower_id="n2", conflict_term=1, conflict_index=3)
    env = Envelope("n2", "n1", reply)
    assert _roundtrip(env) == env


def test_client_request_roundtrips():
    framed = encode_client_request("c1", 4, "put", "k", "v")
    length = int.from_bytes(framed[:4], "big")
    body = framed[4 : 4 + length]
    got = decode_client_request(body)
    assert got == ClientCommand(client_id="c1", seq=4, op="put", key="k", value="v")


def test_client_reply_roundtrips():
    framed = encode_client_reply(ok=True, value="v", existed=True, leader_hint="n2")
    length = int.from_bytes(framed[:4], "big")
    body = framed[4 : 4 + length]
    got = decode_client_reply(body)
    assert got == {"ok": True, "value": "v", "existed": True, "leader_hint": "n2", "error": None}
