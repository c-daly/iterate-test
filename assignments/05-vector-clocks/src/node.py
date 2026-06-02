"""Nodes that produce events and exchange messages.

A :class:`Node` owns a current :class:`~src.vector_clock.VectorClock` and an
:class:`~src.events.EventLog`.  Every action (a local event, a send, or a
receive) advances the clock and records an :class:`~src.events.Event`.  A
:class:`Message` is the on-the-wire envelope: it carries the senders clock so
the receiver can merge it.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from src.events import Event, EventLog
from src.vector_clock import VectorClock

# Monotonic logical timestamp source shared across nodes, so events get a
# stable total tie-break order independent of wall-clock resolution.
_TS = itertools.count(1)


def _next_timestamp() -> float:
    return float(next(_TS))


@dataclass
class Message:
    """An in-flight message from one node to another.

    ``clock`` is the senders vector clock at send time; the receiver merges
    it to inherit the senders causal history.
    """

    sender_id: str
    clock: VectorClock
    data: Any = None


class Node:
    """A participant in the distributed system."""

    def __init__(self, node_id: str, num_nodes: int, event_log: EventLog):
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.event_log = event_log
        self.clock = VectorClock(node_id, num_nodes)

    def local_event(self, data: Any = None) -> Event:
        """Advance the clock for a purely local event and record it."""
        self.clock = self.clock.increment()
        event = Event(
            node_id=self.node_id,
            event_type="local",
            clock=self.clock,
            timestamp=_next_timestamp(),
            data=data,
        )
        self.event_log.record(event)
        return event

    def send(self, data: Any = None) -> tuple[Event, "Message"]:
        """Record a send event and return it with the outgoing message."""
        self.clock = self.clock.increment()
        event = Event(
            node_id=self.node_id,
            event_type="send",
            clock=self.clock,
            timestamp=_next_timestamp(),
            data=data,
        )
        self.event_log.record(event)
        message = Message(sender_id=self.node_id, clock=self.clock, data=data)
        return event, message

    def receive(self, message: "Message") -> Event:
        """Merge the senders clock, record a receive event, return it."""
        self.clock = self.clock.merge(message.clock)
        event = Event(
            node_id=self.node_id,
            event_type="receive",
            clock=self.clock,
            timestamp=_next_timestamp(),
            data=message.data,
        )
        self.event_log.record(event)
        return event
