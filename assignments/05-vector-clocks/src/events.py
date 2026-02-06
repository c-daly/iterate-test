from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .vector_clock import VectorClock


@dataclass
class Event:
    node_id: str
    event_type: str  # "local", "send", "receive"
    clock: VectorClock
    timestamp: float
    data: Any = None


class EventLog:
    """Log of events supporting causal ordering and conflict detection."""

    def __init__(self) -> None:
        self._events: list[Event] = []

    def record(self, event: Event) -> None:
        self._events.append(event)

    def causal_order(self) -> list[Event]:
        """Return events in causal order using topological sort."""
        events = list(self._events)
        # Topological sort based on happens_before
        # Use a stable sort: if a happens_before b, a comes first
        # For concurrent events, preserve insertion order (stable sort)
        n = len(events)
        # Simple approach: repeatedly find events with no predecessors
        result: list[Event] = []
        remaining = list(range(n))

        while remaining:
            # Find an event that no other remaining event happens-before
            # i.e., no remaining event is a predecessor
            found = -1
            for idx in remaining:
                is_minimal = True
                for other_idx in remaining:
                    if other_idx == idx:
                        continue
                    if events[other_idx].clock.happens_before(events[idx].clock):
                        is_minimal = False
                        break
                if is_minimal:
                    found = idx
                    break
            if found == -1:
                # All remaining are concurrent; add in original order
                for idx in remaining:
                    result.append(events[idx])
                break
            result.append(events[found])
            remaining.remove(found)

        return result

    def concurrent_pairs(self) -> list[tuple[Event, Event]]:
        """Return all pairs of concurrent events."""
        pairs: list[tuple[Event, Event]] = []
        n = len(self._events)
        for i in range(n):
            for j in range(i + 1, n):
                if self._events[i].clock.is_concurrent(self._events[j].clock):
                    pairs.append((self._events[i], self._events[j]))
        return pairs

    def find_conflicts(self, key: str) -> list[tuple[Event, Event]]:
        """Find concurrent writes to the same key."""
        # Filter events that have data with matching key
        key_events = [
            e for e in self._events
            if isinstance(e.data, dict) and e.data.get("key") == key
        ]
        conflicts: list[tuple[Event, Event]] = []
        for i in range(len(key_events)):
            for j in range(i + 1, len(key_events)):
                if key_events[i].clock.is_concurrent(key_events[j].clock):
                    conflicts.append((key_events[i], key_events[j]))
        return conflicts
