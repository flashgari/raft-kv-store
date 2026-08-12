"""A from-scratch implementation of the Raft consensus protocol.

Architecture note: the algorithm (this package) has zero I/O. `RaftNode`
is a pure state machine -- every public method takes the current
simulated time and/or an inbound message, and *returns* a list of
outbound messages rather than sending them. This is what makes the
protocol testable: `tests/` drives thousands of randomized fault-injection
trials against an in-process simulated network in well under a second,
with no real sleeping, no real sockets, and fully deterministic replay
from a random seed. `transport/` is the thin, separate layer that wires
this state machine up to real asyncio TCP sockets for the live demo.
"""
