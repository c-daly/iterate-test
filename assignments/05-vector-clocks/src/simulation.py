"""Simulation: orchestrates multiple Nodes plus per-recipient message queues."""
from __future__ import annotations

from collections import deque
from typing import Any

from .events import Event, EventLog
from .node import Message, Node


class Simulation:
    """Coordinates message passing between named Nodes for a shared EventLog."""

    def __init__(self, node_ids: list[str]) -> None:
        self.event_log = EventLog()
        n = len(node_ids)
        self.nodes: dict[str, Node] = {
            nid: Node(nid, n, self.event_log) for nid in node_ids
        }
        # FIFO message queues, one per recipient. deque pops in O(1).
        self.inboxes: dict[str, deque[Message]] = {nid: deque() for nid in node_ids}

    def local_event(self, node_id: str, data: Any = None) -> Event:
        """Run a local event on node_id."""
        return self.nodes[node_id].local_event(data)

    def send_message(
        self, from_id: str, to_id: str, data: Any = None
    ) -> Event:
        """Send a message from from_id to to_id; returns the send Event."""
        ev, msg = self.nodes[from_id].send(data)
        self.inboxes[to_id].append(msg)
        return ev

    def deliver_message(self, to_id: str) -> Event:
        """Pop the oldest queued message and deliver it; returns receive Event."""
        if not self.inboxes[to_id]:
            raise IndexError(f"no pending messages for {to_id!r}")
        msg: Message = self.inboxes[to_id].popleft()
        return self.nodes[to_id].receive(msg)

    def get_log(self) -> EventLog:
        """Return the shared EventLog."""
        return self.event_log

    def get_history(self) -> list[Event]:
        """Return events in causal order (alias for log.causal_order())."""
        return self.event_log.causal_order()
