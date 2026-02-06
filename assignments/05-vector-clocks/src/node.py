from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .events import Event, EventLog
from .vector_clock import VectorClock


@dataclass
class Message:
    sender_id: str
    receiver_id: str
    clock: VectorClock
    data: Any = None


class Node:
    """A node in a distributed system with a vector clock."""

    def __init__(self, node_id: str, num_nodes: int, event_log: EventLog) -> None:
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.event_log = event_log
        self.clock = VectorClock(node_id, num_nodes)

    def local_event(self, data: Any = None) -> Event:
        """Perform a local event: increment clock and record."""
        self.clock = self.clock.increment()
        event = Event(
            node_id=self.node_id,
            event_type="local",
            clock=self.clock,
            timestamp=time.time(),
            data=data,
        )
        self.event_log.record(event)
        return event

    def send(self, data: Any = None) -> tuple[Event, Message]:
        """Send a message: increment clock, create event and message."""
        self.clock = self.clock.increment()
        event = Event(
            node_id=self.node_id,
            event_type="send",
            clock=self.clock,
            timestamp=time.time(),
            data=data,
        )
        self.event_log.record(event)
        message = Message(
            sender_id=self.node_id,
            receiver_id="",  # Will be set by simulation or caller
            clock=self.clock,
            data=data,
        )
        return event, message

    def receive(self, message: Message) -> Event:
        """Receive a message: merge clocks and record event."""
        self.clock = self.clock.merge(message.clock)
        event = Event(
            node_id=self.node_id,
            event_type="receive",
            clock=self.clock,
            timestamp=time.time(),
            data=message.data,
        )
        self.event_log.record(event)
        return event
