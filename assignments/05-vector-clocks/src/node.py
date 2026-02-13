from dataclasses import dataclass
from typing import Any
import time
from vector_clock import VectorClock
from events import Event, EventLog


@dataclass
class Message:
    sender_id: str
    clock: VectorClock
    data: Any = None


class Node:
    def __init__(self, node_id: str, num_nodes: int, event_log: EventLog):
        self.node_id = node_id
        self.clock = VectorClock(node_id, num_nodes)
        self.event_log = event_log

    def local_event(self, data: Any = None) -> Event:
        self.clock.increment()
        event = Event(
            node_id=self.node_id, event_type="local",
            clock=self.clock.copy(), timestamp=time.time(), data=data
        )
        self.event_log.record(event)
        return event

    def send(self, data: Any = None) -> tuple[Event, Message]:
        self.clock.increment()
        event = Event(
            node_id=self.node_id, event_type="send",
            clock=self.clock.copy(), timestamp=time.time(), data=data
        )
        self.event_log.record(event)
        msg = Message(sender_id=self.node_id, clock=self.clock.copy(), data=data)
        return event, msg

    def receive(self, message: Message) -> Event:
        self.clock.merge(message.clock)
        event = Event(
            node_id=self.node_id, event_type="receive",
            clock=self.clock.copy(), timestamp=time.time(), data=message.data
        )
        self.event_log.record(event)
        return event
