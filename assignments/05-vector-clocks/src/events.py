"""Event records and the EventLog for causal-order analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from .vector_clock import VectorClock


@dataclass
class Event:
    """A single observable event recorded by some node.

    Attributes:
        node_id: Origin node.
        event_type: One of 'local', 'send', or 'receive'.
        clock: Snapshot of the origin node's vector clock after the event.
        timestamp: Wall-clock time (unix seconds) for tie-breaking ordering.
        data: Optional payload (often {'key': ..., 'value': ...}).
    """

    node_id: str
    event_type: str
    clock: VectorClock
    timestamp: float
    data: Any = None

    # Identity-based equality/hash so events stay set-able even when their
    # data payload is unhashable (e.g. dicts).
    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


@dataclass
class EventLog:
    """Append-only event log with causal-order and conflict utilities."""

    events: list = field(default_factory=list)

    def record(self, event: Event) -> None:
        """Append event to the log."""
        self.events.append(event)

    def causal_order(self) -> list:
        """Return events topologically sorted by happens-before.

        Wall-clock timestamp breaks ties between concurrent or otherwise
        unordered events, giving a stable, deterministic ordering.
        """
        ordered: list = []
        for ev in sorted(self.events, key=lambda e: e.timestamp):
            inserted = False
            for i, existing in enumerate(ordered):
                if ev.clock.happens_before(existing.clock):
                    ordered.insert(i, ev)
                    inserted = True
                    break
            if not inserted:
                ordered.append(ev)
        return ordered

    def concurrent_pairs(self) -> list:
        """Return every unordered pair of events whose clocks are concurrent."""
        return [
            (a, b)
            for a, b in combinations(self.events, 2)
            if a.clock.is_concurrent(b.clock)
        ]

    def find_conflicts(self, key: str) -> list:
        """Return concurrent pairs that both touch key in their data dict."""
        touching = [
            ev
            for ev in self.events
            if isinstance(ev.data, dict) and ev.data.get("key") == key
        ]
        return [
            (a, b)
            for a, b in combinations(touching, 2)
            if a.clock.is_concurrent(b.clock)
        ]
