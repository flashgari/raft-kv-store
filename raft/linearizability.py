"""A from-scratch linearizability checker (Wing & Gong 1993 / the
algorithm underlying Jepsen's Knossos checker): given a history of
concurrent client operations, each with an observed [start, end] interval
and result, decide whether SOME sequential order of all operations --
consistent with real-time (an operation that finished before another
started must come first) -- reproduces every observed result against the
KV register's sequential specification.

This is deliberately independent of `kvstore.KVStateMachine`: the
specification (`_apply_spec` below) reimplements put/get/delete semantics
from scratch rather than calling the system under test, because a checker
that validated a system by re-running the system's own code would pass
vacuously on a bug in that shared code. The two are compared, not reused.

Complexity note: linearizability checking is NP-hard in general (Gibbons
& Korach, 1997) -- this implementation uses the classic exponential
backtracking search with memoization on (remaining operations, abstract
state), which is what makes it tractable for the history sizes actually
produced by tests/test_linearizability.py (a handful of concurrent
clients, a few operations each -- dozens of operations, not thousands).
It is a correctness-testing tool, not a claim of a scalable production
checker.
"""

from __future__ import annotations

from dataclasses import dataclass

from .kvstore import Operation, Reply


def _apply_spec(state: dict, op: Operation) -> tuple[dict, Reply]:
    if op.op == "put":
        new_state = dict(state)
        new_state[op.key] = op.value
        return new_state, Reply(ok=True, value=op.value)
    if op.op == "delete":
        existed = op.key in state
        new_state = dict(state)
        new_state.pop(op.key, None)
        return new_state, Reply(ok=True, existed=existed)
    if op.op == "get":
        return state, Reply(ok=op.key in state, value=state.get(op.key))
    raise ValueError(f"unknown op {op.op!r}")


@dataclass
class LinearizabilityResult:
    linearizable: bool
    witness_order: tuple[int, ...] | None  # operation indices, if linearizable
    states_explored: int


def check_linearizable(history: list[Operation]) -> LinearizabilityResult:
    n = len(history)
    memo_fail: set[tuple[frozenset, tuple]] = set()
    states_explored = 0
    witness: list[int] = []

    def state_key(state: dict) -> tuple:
        return tuple(sorted(state.items()))

    def recurse(state: dict, remaining: frozenset) -> bool:
        nonlocal states_explored
        if not remaining:
            return True
        key = (remaining, state_key(state))
        if key in memo_fail:
            return False
        states_explored += 1

        for i in sorted(remaining):
            op = history[i]
            # Real-time-order pruning: op cannot be linearized next if some
            # OTHER still-pending operation is known to have to precede it
            # (it completed, in real time, before op even started).
            blocked = any(j != i and history[j].end_ms <= op.start_ms for j in remaining)
            if blocked:
                continue
            new_state, expected = _apply_spec(state, op)
            if expected != op.reply:
                continue
            witness.append(i)
            if recurse(new_state, remaining - {i}):
                return True
            witness.pop()

        memo_fail.add(key)
        return False

    ok = recurse({}, frozenset(range(n)))
    return LinearizabilityResult(
        linearizable=ok,
        witness_order=tuple(witness) if ok else None,
        states_explored=states_explored,
    )
