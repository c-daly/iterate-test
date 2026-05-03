"""Distributed simulation node with vector-clock-stamped events."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Tuple

from src.events import Event, EventLog
from src.vector_clock import VectorClock


@dataclass
class Message:
    """Inter-node message carrying the senders vector clock and payload."""
    clock: VectorClock
    data: Any = None
    sender_id: str = ""


class Node:
    """A node with its own vector clock; mutates the clock as events occur."""

    def __init__(self, node_id: str, num_nodes: int, event_log: EventLog) -> None:
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.clock = VectorClock(node_id, num_nodes)
        self.event_log = event_log
        self._seq = 0  # monotonic per-node sequence for tie-break stability

    def _next_timestamp(self) -> float:
        # Use wall clock plus a tiny per-node offset for monotonicity.
        ts = time.time() + self._seq * 1e-4
        self._seq += 1
        return ts

    def local_event(self, data: Any = None) -> Event:
        self.clock = self.clock.increment()
        ev = Event(
            node_id=self.node_id,
            event_type="local",
            clock=self.clock,
            timestamp=self._next_timestamp(),
            data=data,
        )
        self.event_log.record(ev)
        return ev

    def send(self, data: Any = None) -> Tuple[Event, Message]:
        self.clock = self.clock.increment()
        ev = Event(
            node_id=self.node_id,
            event_type="send",
            clock=self.clock,
            timestamp=self._next_timestamp(),
            data=data,
        )
        self.event_log.record(ev)
        msg = Message(clock=self.clock, data=data, sender_id=self.node_id)
        return ev, msg

    def receive(self, message: Message) -> Event:
        self.clock = self.clock.merge(message.clock)
        ev = Event(
            node_id=self.node_id,
            event_type="receive",
            clock=self.clock,
            timestamp=self._next_timestamp(),
            data=message.data,
        )
        self.event_log.record(ev)
        return ev
