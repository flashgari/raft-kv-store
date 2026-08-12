"""The replicated log. 1-indexed, with a sentinel at index 0 (term 0) so
`prev_log_index=0` (an empty log) always has a well-defined term to compare
against -- this removes a whole class of off-by-one special-casing from
the AppendEntries consistency check in node.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .messages import LogEntry

_SENTINEL = LogEntry(term=0, index=0, command=None)


@dataclass
class RaftLog:
    entries: list[LogEntry] = field(default_factory=lambda: [_SENTINEL])

    def __len__(self) -> int:
        return len(self.entries) - 1  # exclude the sentinel

    @property
    def last_index(self) -> int:
        return self.entries[-1].index

    @property
    def last_term(self) -> int:
        return self.entries[-1].term

    def term_at(self, index: int) -> int | None:
        if index < 0 or index >= len(self.entries):
            return None
        return self.entries[index].term

    def entry_at(self, index: int) -> LogEntry | None:
        if index <= 0 or index >= len(self.entries):
            return None
        return self.entries[index]

    def slice_from(self, index: int) -> tuple[LogEntry, ...]:
        """Entries at index..last_index, inclusive. Empty if index > last_index."""
        if index > self.last_index:
            return ()
        return tuple(self.entries[index:])

    def append(self, entry: LogEntry) -> None:
        assert entry.index == self.last_index + 1, f"append out of order: {entry.index} != {self.last_index + 1}"
        self.entries.append(entry)

    def truncate_after(self, index: int) -> None:
        """Delete all entries after `index` (used when a conflict is found)."""
        del self.entries[index + 1 :]

    def merge_from_leader(self, prev_log_index: int, new_entries: tuple[LogEntry, ...]) -> bool:
        """Apply AppendEntries' entries list starting at prev_log_index + 1.

        Implements the exact conflict rule from the paper (§5.3): for each
        new entry, if the log already has a DIFFERENT entry at that index
        (same index, different term), truncate from there on and then
        append. If the log already has the SAME entry, skip it -- this
        matters because re-appending an already-matching suffix must be a
        no-op, not a truncate-and-replace, or a delayed/duplicated
        AppendEntries RPC could erase log entries a later RPC already
        confirmed are correct.
        """
        insert_at = prev_log_index + 1
        for offset, entry in enumerate(new_entries):
            idx = insert_at + offset
            existing = self.entry_at(idx)
            if existing is not None:
                if existing.term == entry.term:
                    continue  # already have this exact entry; not a conflict
                self.truncate_after(idx - 1)
                self.append(entry)
            else:
                self.append(entry)
        return True

    def conflict_info_for(self, prev_log_index: int) -> tuple[int | None, int | None]:
        """When an AppendEntries consistency check fails, compute the fast-backup
        hint (conflict_term, conflict_index) a follower reports back (§5.3 /
        the Raft student guide's "fast backup" scheme)."""
        if prev_log_index > self.last_index:
            # Follower's log is just too short.
            return None, self.last_index + 1
        conflict_term = self.term_at(prev_log_index)
        # Walk back to the first entry of that conflicting term.
        first_index_of_term = prev_log_index
        while first_index_of_term > 1 and self.term_at(first_index_of_term - 1) == conflict_term:
            first_index_of_term -= 1
        return conflict_term, first_index_of_term


def candidate_log_is_up_to_date(
    candidate_last_term: int, candidate_last_index: int, voter_last_term: int, voter_last_index: int
) -> bool:
    """§5.4.1's actual voting rule, phrased from the voter's side: grant a
    vote only if the CANDIDATE's log is at least as up to date as the
    VOTER's own log -- i.e. deny the vote when the voter's log is strictly
    more up to date than the candidate's. This is deliberately a free
    function taking both sides explicitly rather than a `RaftLog` instance
    method: an instance method naturally reads as "is MY log >= the other
    one", which is exactly backwards from what the voting rule needs. An
    earlier version of this code was a `RaftLog.is_at_least_as_up_to_date_as`
    instance method called as `self.log.is_at_least_as_up_to_date_as(candidate...)`
    from `_on_request_vote` -- it silently computed "is the voter's log at
    least as up to date as the candidate's", the reverse of the actual
    rule, and a candidate with a strictly staler log could still win votes.
    Caught by `tests/test_node_unit.py::test_rejects_candidate_with_less_up_to_date_log`
    (see README, "Bugs found and fixed").
    """
    if candidate_last_term != voter_last_term:
        return candidate_last_term > voter_last_term
    return candidate_last_index >= voter_last_index
