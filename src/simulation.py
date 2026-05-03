"""Multi-node simulation orchestrating local events and message passing."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, List

from src.events import Event, EventLog
from src.node import Message, Node


class Simulation:
    """Coordinates a fixed set of nodes and a per-recipient message inbox."""

    def __init__(self, node_ids: List[str]) -> None:
        self.event_log = EventLog()
        self.nodes: Dict[str, Node] = {
            nid: Node(nid, len(node_ids), self.event_log) for nid in node_ids
        }
        # Per-recipient FIFO inbox of pending messages.
        self._inboxes: Dict[str, Deque[Message]] = defaultdict(deque)

    # ------------------------------------------------------------------
    # Driver API.
    # ------------------------------------------------------------------

    def local_event(self, node_id: str, data: Any = None) -> Event:
        return self.nodes[node_id].local_event(data)

    def send_message(self, from_id: str, to_id: str, data: Any = None) -> Event:
        if to_id not in self.nodes:
            raise KeyError(f"unknown recipient: {to_id}")
        ev, msg = self.nodes[from_id].send(data)
        self._inboxes[to_id].append(msg)
        return ev

    def deliver_message(self, to_id: str) -> Event:
        inbox = self._inboxes.get(to_id)
        if not inbox:
            raise IndexError(f"no pending messages for {to_id}")
        msg = inbox.popleft()
        return self.nodes[to_id].receive(msg)

    # ------------------------------------------------------------------
    # Inspection.
    # ------------------------------------------------------------------

    def get_log(self) -> EventLog:
        return self.event_log

    def get_history(self) -> List[Event]:
        return list(self.event_log.events)
