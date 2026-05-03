"""Vector clock for tracking causality across distributed nodes.

A VectorClock is an immutable mapping from node-id to a monotonically
increasing integer counter. Operations (increment, merge) return fresh
clocks rather than mutating in place, which keeps clocks safe to store in
events and to use as dictionary or set members.

The internal counter dictionary is keyed by node_id. num_nodes is retained
as documentation of the cluster size; the dict itself is grown lazily as
peers are observed.
"""
from __future__ import annotations

from typing import Mapping


class VectorClock:
    """Immutable vector clock owned by node_id."""

    __slots__ = ("node_id", "num_nodes", "_counters")

    def __init__(
        self,
        node_id: str,
        num_nodes: int,
        _counters: Mapping[str, int] | None = None,
    ) -> None:
        self.node_id = node_id
        self.num_nodes = num_nodes
        self._counters: dict[str, int] = (
            dict(_counters) if _counters is not None else {}
        )

    def get(self, node_id: str) -> int:
        return self._counters.get(node_id, 0)

    def as_dict(self) -> dict[str, int]:
        return dict(self._counters)

    def increment(self) -> VectorClock:
        c = dict(self._counters)
        c[self.node_id] = c.get(self.node_id, 0) + 1
        return VectorClock(self.node_id, self.num_nodes, c)

    def merge(self, other: VectorClock) -> VectorClock:
        m = dict(self._counters)
        for k, v in other._counters.items():
            if v > m.get(k, 0):
                m[k] = v
        m[self.node_id] = m.get(self.node_id, 0) + 1
        return VectorClock(self.node_id, self.num_nodes, m)

    def happens_before(self, other: VectorClock) -> bool:
        keys = set(self._counters) | set(other._counters)
        strictly_less = False
        for k in keys:
            s = self._counters.get(k, 0)
            o = other._counters.get(k, 0)
            if s > o:
                return False
            if s < o:
                strictly_less = True
        return strictly_less

    def is_concurrent(self, other: VectorClock) -> bool:
        return (
            not self.happens_before(other)
            and not other.happens_before(self)
            and self != other
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        keys = set(self._counters) | set(other._counters)
        return all(self._counters.get(k, 0) == other._counters.get(k, 0) for k in keys)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        keys = set(self._counters) | set(other._counters)
        return all(self._counters.get(k, 0) <= other._counters.get(k, 0) for k in keys)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        return self.happens_before(other)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._counters.items())))

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v}" for k, v in sorted(self._counters.items()))
        return f"VectorClock(node_id={self.node_id!r}, {items})"
