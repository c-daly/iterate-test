from __future__ import annotations

from collections import defaultdict
from typing import Any

from .events import Event, EventLog
from .node import Message, Node


class Simulation:
    """Simulation of a distributed system with multiple nodes."""

    def __init__(self, node_ids: list[str]) -> None:
        self._log = EventLog()
        num_nodes = len(node_ids)
        self._nodes: dict[str, Node] = {
            nid: Node(nid, num_nodes, self._log) for nid in node_ids
        }
        self._pending: dict[str, list[Message]] = defaultdict(list)

    def local_event(self, node_id: str, data: Any = None) -> Event:
        return self._nodes[node_id].local_event(data=data)

    def send_message(
        self, from_id: str, to_id: str, data: Any = None
    ) -> Event:
        event, message = self._nodes[from_id].send(data=data)
        message.receiver_id = to_id
        self._pending[to_id].append(message)
        return event

    def deliver_message(self, to_id: str) -> Event:
        if not self._pending[to_id]:
            raise ValueError(f"No pending messages for node {to_id}")
        message = self._pending[to_id].pop(0)
        return self._nodes[to_id].receive(message)

    def get_log(self) -> EventLog:
        return self._log

    def get_history(self) -> list[Event]:
        return self._log.causal_order()
