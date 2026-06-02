"""A multi-node simulation driver.

:class:`Simulation` wires up a set of :class:`~src.node.Node` instances that
share a single :class:`~src.events.EventLog`, and drives them through local
events and (possibly delayed) message delivery.  Messages are queued per
destination so a send and its matching deliver can be interleaved with other
activity -- which is exactly what makes concurrency and conflicts observable.
"""
from __future__ import annotations

from collections import deque
from typing import Any

from src.events import Event, EventLog
from src.node import Message, Node


class Simulation:
    """Coordinate a fixed set of nodes over a shared event log."""

    def __init__(self, node_ids: list[str]):
        self.node_ids = list(node_ids)
        self.log = EventLog()
        num_nodes = len(self.node_ids)
        self.nodes: dict[str, Node] = {
            nid: Node(nid, num_nodes, self.log) for nid in self.node_ids
        }
        # Per-destination FIFO of undelivered messages.
        self._inboxes: dict[str, deque[Message]] = {
            nid: deque() for nid in self.node_ids
        }

    def _node(self, node_id: str) -> Node:
        try:
            return self.nodes[node_id]
        except KeyError:
            raise KeyError(f"unknown node id: {node_id!r}") from None

    def local_event(self, node_id: str, data: Any = None) -> Event:
        """Run a local event on ``node_id``."""
        return self._node(node_id).local_event(data)

    def send_message(
        self, from_id: str, to_id: str, data: Any = None
    ) -> Event:
        """Send from ``from_id`` to ``to_id``; queue it for later delivery.

        Returns the *send* event.  The message sits in the destinations
        inbox until :meth:`deliver_message` is called for that destination.
        """
        if to_id not in self.nodes:
            raise KeyError(f"unknown destination node id: {to_id!r}")
        event, message = self._node(from_id).send(data)
        self._inboxes[to_id].append(message)
        return event

    def deliver_message(self, to_id: str) -> Event:
        """Deliver the oldest queued message to ``to_id``.

        Raises :class:`IndexError` if no message is waiting -- delivering
        nothing would silently no-op and hide ordering bugs.
        """
        node = self._node(to_id)
        inbox = self._inboxes[to_id]
        if not inbox:
            raise IndexError(f"no message queued for node {to_id!r}")
        message = inbox.popleft()
        return node.receive(message)

    def get_log(self) -> EventLog:
        """Return the shared event log."""
        return self.log

    def get_history(self) -> list[Event]:
        """Return all recorded events in a causally consistent order."""
        return self.log.causal_order()
