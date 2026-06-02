"""Events and the event log.

An :class:`Event` is an immutable record of something that happened at a node,
stamped with the :class:`~src.vector_clock.VectorClock` value at that moment.
The :class:`EventLog` collects events and answers causality questions over the
whole set: a causal (topological) ordering, the set of concurrent pairs, and
write/write conflicts on a given key.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.vector_clock import VectorClock


@dataclass
class Event:
    """A single recorded event.

    ``event_type`` is one of ``"local"``, ``"send"`` or ``"receive"``.
    ``data`` is an arbitrary payload; for write events it is conventionally a
    mapping containing ``"key"`` and ``"value"`` so conflicts can be detected.
    Identity-based equality is used (``eq=False``) so distinct events with the
    same field values remain distinguishable inside the log.
    """

    node_id: str
    event_type: str
    clock: VectorClock
    timestamp: float
    data: Any = None

    # Use identity equality/hashing: two writes with the same payload are
    # still two different events and must not collapse in sets/indexing.
    __hash__ = object.__hash__

    def __eq__(self, other):  # noqa: D401 - identity semantics
        return self is other

    def _key(self) -> Any:
        """Return the key this event writes, if any."""
        if isinstance(self.data, dict):
            return self.data.get("key")
        return None


class EventLog:
    """An append-only collection of events with causality queries."""

    def __init__(self) -> None:
        self._events: list[Event] = []

    def record(self, event: Event) -> None:
        """Append ``event`` to the log in arrival order."""
        self._events.append(event)

    @property
    def events(self) -> list[Event]:
        """Events in the order they were recorded."""
        return list(self._events)

    def causal_order(self) -> list[Event]:
        """Return events in a causally consistent (topological) order.

        If ``e1`` happens-before ``e2`` then ``e1`` appears first.  Concurrent
        events are broken by ``timestamp`` and then recording order so the
        result is deterministic.  A stable insertion sort over the
        happens-before relation is used: it is O(n^2) but exact and never
        violates a causal edge.
        """
        ordered: list[Event] = []
        # Seed with a deterministic baseline (timestamp, then record order)
        # so concurrent events keep a stable relative position.
        baseline = sorted(
            enumerate(self._events), key=lambda p: (p[1].timestamp, p[0])
        )
        for _, event in baseline:
            insert_at = len(ordered)
            for i, placed in enumerate(ordered):
                # event must come before any already-placed event it precedes
                if event.clock.happens_before(placed.clock):
                    insert_at = i
                    break
            ordered.insert(insert_at, event)
        return ordered

    def concurrent_pairs(self) -> list[tuple[Event, Event]]:
        """Return every unordered pair of mutually concurrent events."""
        pairs: list[tuple[Event, Event]] = []
        events = self._events
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                a, b = events[i], events[j]
                if a.clock.is_concurrent(b.clock):
                    pairs.append((a, b))
        return pairs

    def find_conflicts(self, key: str) -> list[tuple[Event, Event]]:
        """Return concurrent write pairs that both touch ``key``.

        Two writes conflict when they target the same key and are concurrent
        (neither causally precedes the other) -- the classic last-writer-wins
        ambiguity that vector clocks expose.
        """
        writes = [e for e in self._events if e._key() == key]
        conflicts: list[tuple[Event, Event]] = []
        for i in range(len(writes)):
            for j in range(i + 1, len(writes)):
                a, b = writes[i], writes[j]
                if a.clock.is_concurrent(b.clock):
                    conflicts.append((a, b))
        return conflicts
