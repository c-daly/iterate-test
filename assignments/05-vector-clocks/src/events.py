"""Event records and the EventLog for causal-order analysis."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from .vector_clock import VectorClock


@dataclass
class Event:
    """A single observable event recorded by some node.

    Attributes:
        node_id: Origin node.
        event_type: One of local, send, or receive.
        clock: Snapshot of the origin nodes vector clock after the event.
        timestamp: Wall-clock time (unix seconds) for tie-breaking ordering.
        data: Optional payload (often {key: ..., value: ...}).
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

    events: list[Event] = field(default_factory=list)

    def record(self, event: Event) -> None:
        """Append event to the log."""
        self.events.append(event)

    def causal_order(self) -> list[Event]:
        """Return events in a topological order respecting happens-before.

        The ordering is any topological order respecting causality, with
        wall-clock timestamp used to break ties between concurrent events
        (giving a stable, deterministic result).

        Implementation: Kahns algorithm with indegree counts and a deque,
        seeded from a timestamp-sorted list. Edge build is O(N^2) over event
        pairs (unavoidable without extra structure on clocks), but processing
        is O(N + E) with O(1) deque appends/pops -- no inner ``list.insert``.
        """
        events = sorted(self.events, key=lambda e: e.timestamp)
        n = len(events)
        # Adjacency list: predecessors[i] -> indices that must come before i.
        # Using indices keeps Event objects hashable-by-identity safe.
        successors: list[list[int]] = [[] for _ in range(n)]
        indegree = [0] * n
        for i, j in combinations(range(n), 2):
            # events[i].timestamp <= events[j].timestamp by sort.
            ci = events[i].clock
            cj = events[j].clock
            if ci.happens_before(cj):
                successors[i].append(j)
                indegree[j] += 1
            elif cj.happens_before(ci):
                successors[j].append(i)
                indegree[i] += 1
            # else: concurrent or equal -- no causal edge; timestamp order wins.

        # Kahns algorithm: process in original (timestamp-sorted) index order
        # so concurrent events surface in deterministic wall-clock order.
        ready: deque[int] = deque(i for i in range(n) if indegree[i] == 0)
        ordered: list[Event] = []
        while ready:
            i = ready.popleft()
            ordered.append(events[i])
            for j in successors[i]:
                indegree[j] -= 1
                if indegree[j] == 0:
                    ready.append(j)
        return ordered

    def concurrent_pairs(self) -> list[tuple[Event, Event]]:
        """Return every unordered pair of events whose clocks are concurrent."""
        return [
            (a, b)
            for a, b in combinations(self.events, 2)
            if a.clock.is_concurrent(b.clock)
        ]

    def find_conflicts(self, key: str) -> list[tuple[Event, Event]]:
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
