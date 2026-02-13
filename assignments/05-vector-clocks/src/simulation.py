from typing import Any
from collections import defaultdict, deque
from events import Event, EventLog
from node import Node, Message


class Simulation:
    def __init__(self, node_ids: list[str]):
        self.event_log = EventLog()
        self.nodes = {nid: Node(nid, len(node_ids), self.event_log) for nid in node_ids}
        self._pending = defaultdict(deque)

    def local_event(self, node_id: str, data: Any = None) -> Event:
        return self.nodes[node_id].local_event(data)

    def send_message(self, from_id: str, to_id: str, data: Any = None) -> Event:
        event, msg = self.nodes[from_id].send(data)
        self._pending[to_id].append(msg)
        return event

    def deliver_message(self, to_id: str) -> Event:
        if not self._pending[to_id]:
            raise ValueError(f"No pending messages for {to_id}")
        msg = self._pending[to_id].popleft()
        return self.nodes[to_id].receive(msg)

    def get_log(self) -> EventLog:
        return self.event_log

    def get_history(self) -> list[Event]:
        return self.event_log.causal_order()
