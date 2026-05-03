"""Node abstraction: produces local/send/receive events with vector clocks."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .events import Event, EventLog
from .vector_clock import VectorClock


@dataclass
class Message:
    """In-flight message between two nodes.

    Carries the senders vector clock so the receiver can merge.
    """

    sender: str
    clock: VectorClock
    data: Any = None


class Node:
    """A node in the distributed system.

    Each node owns a vector clock and writes events to a shared EventLog.
    """

    def __init__(self, node_id: str, num_nodes: int, event_log: EventLog) -> None:
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.event_log = event_log
        self.clock = VectorClock(node_id, num_nodes)

    def local_event(self, data: Any = None) -> Event:
        """Record a purely local event; bumps the local clock."""
        self.clock = self.clock.increment()
        ev = Event(self.node_id, "local", self.clock, time.time(), data)
        self.event_log.record(ev)
        return ev

    def send(self, data: Any = None) -> tuple[Event, Message]:
        """Record a send event and return (event, message_to_dispatch)."""
        self.clock = self.clock.increment()
        snapshot = self.clock  # immutable; safe to share with the message
        ev = Event(self.node_id, "send", snapshot, time.time(), data)
        self.event_log.record(ev)
        msg = Message(sender=self.node_id, clock=snapshot, data=data)
        return ev, msg

    def receive(self, message: Message) -> Event:
        """Merge incoming clock, record a receive event, return it."""
        self.clock = self.clock.merge(message.clock)
        ev = Event(
            self.node_id, "receive", self.clock, time.time(), message.data
        )
        self.event_log.record(ev)
        return ev
