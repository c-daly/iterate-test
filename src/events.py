"""Event log with causal ordering and conflict detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Any, List, Tuple

from src.vector_clock import VectorClock


@dataclass
class Event:
    node_id: str
    event_type: str  # "local", "send", "receive"
    clock: VectorClock
    timestamp: float
    data: Any = None

    def __hash__(self) -> int:  # identity-based for use as dict/set keys
        return id(self)

    def __eq__(self, other: object) -> bool:  # identity equality
        return self is other


@dataclass
class EventLog:
    events: List[Event] = field(default_factory=list)

    def record(self, event: Event) -> None:
        self.events.append(event)

    def causal_order(self) -> List[Event]:
        """Return events sorted respecting happens-before; ties by (timestamp, node_id)."""
        events = list(self.events)
        n = len(events)
        succ: List[List[int]] = [[] for _ in range(n)]
        indeg = [0] * n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if events[i].clock.happens_before(events[j].clock):
                    succ[i].append(j)
                    indeg[j] += 1

        ready: list = []
        for i in range(n):
            if indeg[i] == 0:
                heappush(ready, (events[i].timestamp, events[i].node_id, i))

        ordered: List[Event] = []
        while ready:
            _, _, i = heappop(ready)
            ordered.append(events[i])
            for j in succ[i]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    heappush(ready, (events[j].timestamp, events[j].node_id, j))

        if len(ordered) != n:
            raise RuntimeError("causal_order: cycle in happens-before graph")
        return ordered

    def concurrent_pairs(self) -> List[Tuple[Event, Event]]:
        events = self.events
        out: List[Tuple[Event, Event]] = []
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                if events[i].clock.is_concurrent(events[j].clock):
                    out.append((events[i], events[j]))
        return out

    def find_conflicts(self, key: str) -> List[Tuple[Event, Event]]:
        """Concurrent event pairs whose data is a dict containing key."""
        out: List[Tuple[Event, Event]] = []
        relevant = [
            e for e in self.events if isinstance(e.data, dict) and key in e.data
        ]
        for i in range(len(relevant)):
            for j in range(i + 1, len(relevant)):
                if relevant[i].clock.is_concurrent(relevant[j].clock):
                    out.append((relevant[i], relevant[j]))
        return out
