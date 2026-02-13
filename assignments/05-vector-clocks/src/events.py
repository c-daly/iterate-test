from dataclasses import dataclass
from typing import Any
from vector_clock import VectorClock


@dataclass
class Event:
    node_id: str
    event_type: str  # 'local', 'send', 'receive'
    clock: VectorClock
    timestamp: float
    data: Any = None


class EventLog:
    def __init__(self):
        self._events = []

    def record(self, event: Event) -> None:
        self._events.append(event)

    def causal_order(self) -> list[Event]:
        return sorted(self._events, key=lambda e: (sum(e.clock.clock.values()), e.timestamp))

    def concurrent_pairs(self) -> list[tuple[Event, Event]]:
        pairs = []
        for i in range(len(self._events)):
            for j in range(i + 1, len(self._events)):
                if self._events[i].clock.is_concurrent(self._events[j].clock):
                    pairs.append((self._events[i], self._events[j]))
        return pairs

    def find_conflicts(self, key: str) -> list[tuple[Event, Event]]:
        relevant = [
            e for e in self._events
            if e.data == key or (isinstance(e.data, dict) and key in e.data)
        ]
        conflicts = []
        for i in range(len(relevant)):
            for j in range(i + 1, len(relevant)):
                if relevant[i].clock.is_concurrent(relevant[j].clock):
                    conflicts.append((relevant[i], relevant[j]))
        return conflicts
