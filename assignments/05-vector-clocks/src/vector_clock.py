"""Vector clocks for tracking causality in a distributed system.

A vector clock maps each node id to a monotonically increasing counter.
Comparing two clocks lets us decide whether one event causally precedes
another (``happens_before``) or whether the two events are ``concurrent``
(neither precedes the other).

All mutating operations return a *new* ``VectorClock`` -- instances are
treated as immutable snapshots so that events can safely hold references to
the clock value at the moment they occurred.
"""
from __future__ import annotations


class VectorClock:
    """A vector clock owned by ``node_id`` over ``num_nodes`` participants.

    The component identities default to the string indices ``"0"`` ..
    ``"<num_nodes-1>"``, plus an entry for ``node_id`` itself.  This keeps the
    class usable both with positional node ids and with explicit string ids:
    whenever a clock from another node is merged in, any unknown components are
    absorbed, so the set of tracked components grows to cover every node that
    has ever been observed.
    """

    def __init__(self, node_id: str, num_nodes: int):
        self.node_id = node_id
        # Seed exactly ``num_nodes`` components.  ``node_id`` is always one of
        # them; the remaining slots are filled with index labels ("0", "1",
        # ...) skipping any that would collide with ``node_id``.  This keeps
        # ``len(clock) == num_nodes`` whether node ids are numeric indices or
        # arbitrary string labels.  Components for nodes first seen via a
        # ``merge`` are absorbed on demand (missing keys read as 0).
        clock: dict[str, int] = {node_id: 0}
        i = 0
        while len(clock) < num_nodes:
            label = str(i)
            if label != node_id:
                clock[label] = 0
            i += 1
        self.clock: dict[str, int] = clock
        # Remember the declared size so derived clocks stay consistent.
        self._num_nodes = num_nodes

    # -- internal helpers ------------------------------------------------

    def _clone_with(self, clock: dict[str, int]) -> "VectorClock":
        new = VectorClock.__new__(VectorClock)
        new.node_id = self.node_id
        new.clock = dict(clock)
        new._num_nodes = self._num_nodes
        return new

    # -- mutating operations (return new clocks) -------------------------

    def increment(self) -> "VectorClock":
        """Return a copy with this nodes own counter advanced by one."""
        updated = dict(self.clock)
        updated[self.node_id] = updated.get(self.node_id, 0) + 1
        return self._clone_with(updated)

    def merge(self, other: "VectorClock") -> "VectorClock":
        """Merge ``other`` into this clock: element-wise max, then increment.

        This is the ``receive`` operation: we take the supremum of the two
        clocks (so we observe everything the sender had seen) and then bump
        our own component to mark the receive event itself.
        """
        keys = set(self.clock) | set(other.clock)
        merged = {
            k: max(self.clock.get(k, 0), other.clock.get(k, 0)) for k in keys
        }
        merged[self.node_id] = merged.get(self.node_id, 0) + 1
        return self._clone_with(merged)

    # -- causality queries ----------------------------------------------

    def happens_before(self, other: "VectorClock") -> bool:
        """True iff this event causally precedes ``other`` (strict).

        Strict partial order: every component is <= the others and at
        least one is strictly <.  Irreflexive (an event never happens before
        itself) and transitive by construction.
        """
        strictly_less = False
        for k, mine in self.clock.items():
            theirs = other.clock.get(k, 0)
            if mine > theirs:
                return False
            if mine < theirs:
                strictly_less = True
        if not strictly_less:
            for k, theirs in other.clock.items():
                if theirs > 0 and k not in self.clock:
                    strictly_less = True
                    break
        return strictly_less

    def is_concurrent(self, other: "VectorClock") -> bool:
        """True iff neither clock happens-before the other and they differ."""
        greater = False
        less = False
        for k, mine in self.clock.items():
            theirs = other.clock.get(k, 0)
            if mine > theirs:
                greater = True
            elif mine < theirs:
                less = True
            if greater and less:
                return True
        if not (greater and less):
            for k, theirs in other.clock.items():
                if k not in self.clock and theirs > 0:
                    less = True
                    if greater:
                        return True
        return False

    # -- comparison operators -------------------------------------------

    def __eq__(self, other) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        for k, v in self.clock.items():
            if v != other.clock.get(k, 0):
                return False
        for k, v in other.clock.items():
            if v > 0 and k not in self.clock:
                return False
        return True

    def __hash__(self):
        # Hash over the non-zero components so equal clocks hash equally
        # regardless of which zero-valued keys each happens to carry.
        items = tuple(sorted((k, v) for k, v in self.clock.items() if v))
        return hash(items)

    def __le__(self, other) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        for k, mine in self.clock.items():
            if mine > other.clock.get(k, 0):
                return False
        return True

    def __lt__(self, other) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        return self.happens_before(other)

    def __repr__(self) -> str:
        body = ", ".join(
            f"{k}:{v}" for k, v in sorted(self.clock.items())
        )
        return f"VectorClock({self.node_id}, {{{body}}})"
