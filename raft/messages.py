"""RPC and timer message types.

Every message is an immutable dataclass. `Envelope` wraps a payload with
routing information (`src`, `dst`) so the pure `RaftNode` can hand a list
of outbound envelopes back to whatever transport (simulated or real) is
driving it, without knowing anything about sockets or event loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass(frozen=True)
class LogEntry:
    term: int
    index: int
    command: object  # opaque to Raft; interpreted by the state machine layer


@dataclass(frozen=True)
class RequestVoteArgs:
    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int


@dataclass(frozen=True)
class RequestVoteReply:
    term: int
    vote_granted: bool
    voter_id: str


@dataclass(frozen=True)
class AppendEntriesArgs:
    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: tuple[LogEntry, ...]
    leader_commit: int


@dataclass(frozen=True)
class AppendEntriesReply:
    term: int
    success: bool
    follower_id: str
    # "Fast backup" optimization (Raft paper §5.3 / student guide): lets the
    # leader skip nextIndex back to the start of the conflicting term in one
    # round trip instead of decrementing by one and retrying per entry.
    conflict_term: int | None = None
    conflict_index: int | None = None
    match_index: int | None = None  # only meaningful when success=True


RpcPayload = Union[RequestVoteArgs, RequestVoteReply, AppendEntriesArgs, AppendEntriesReply]


@dataclass(frozen=True)
class Envelope:
    src: str
    dst: str
    payload: RpcPayload
    # Simulated/real send time in ms, filled in by the transport layer;
    # used by the simulated network to model delay/reordering.
    sent_at_ms: float = field(default=0.0)
