"""Vector clock for tracking causality across nodes."""
from __future__ import annotations

from typing import Dict, Optional


class VectorClock:
    """Vector clock indexed by node id.

    Treated as immutable: ``increment`` and ``merge`` return new instances
    rather than mutating in place.
    """

    __slots__ = ("node_id", "counters")

    def __init__(
        self,
        node_id: str,
        num_nodes: int,
        counters: Optional[Dict[str, int]] = None,
    ) -> None:
        self.node_id = node_id
        if counters is None:
            # Seed with the local node id, then pad with placeholder ids
            # "0", "1", ... until we have ``num_nodes`` entries.
            self.counters = {node_id: 0}
            i = 0
            while len(self.counters) < num_nodes:
                placeholder = str(i)
                if placeholder not in self.counters:
                    self.counters[placeholder] = 0
                i += 1
        else:
            self.counters = dict(counters)
            if node_id not in self.counters:
                self.counters[node_id] = 0

    # ------------------------------------------------------------------
    # Construction helpers.
    # ------------------------------------------------------------------

    def _clone_with(self, counters: Dict[str, int]) -> "VectorClock":
        new = VectorClock.__new__(VectorClock)
        new.node_id = self.node_id
        new.counters = counters
        return new

    # ------------------------------------------------------------------
    # Core operations.
    # ------------------------------------------------------------------

    def increment(self) -> "VectorClock":
        new_counters = dict(self.counters)
        new_counters[self.node_id] = new_counters.get(self.node_id, 0) + 1
        return self._clone_with(new_counters)

    def merge(self, other: "VectorClock") -> "VectorClock":
        keys = set(self.counters) | set(other.counters)
        merged = {
            k: max(self.counters.get(k, 0), other.counters.get(k, 0))
            for k in keys
        }
        merged[self.node_id] = merged.get(self.node_id, 0) + 1
        return self._clone_with(merged)

    # ------------------------------------------------------------------
    # Causality predicates.
    # ------------------------------------------------------------------

    def _le_pointwise(self, other: "VectorClock") -> bool:
        keys = set(self.counters) | set(other.counters)
        return all(
            self.counters.get(k, 0) <= other.counters.get(k, 0) for k in keys
        )

    def happens_before(self, other: "VectorClock") -> bool:
        if not isinstance(other, VectorClock):
            raise TypeError(
                f"happens_before expected VectorClock, got {type(other).__name__}"
            )
        return self._le_pointwise(other) and self != other

    def is_concurrent(self, other: "VectorClock") -> bool:
        if not isinstance(other, VectorClock):
            raise TypeError(
                f"is_concurrent expected VectorClock, got {type(other).__name__}"
            )
        if self == other:
            return False
        return not self.happens_before(other) and not other.happens_before(self)

    # ------------------------------------------------------------------
    # Rich comparisons.
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        keys = set(self.counters) | set(other.counters)
        return all(
            self.counters.get(k, 0) == other.counters.get(k, 0) for k in keys
        )

    def __le__(self, other: "VectorClock") -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        return self._le_pointwise(other)

    def __lt__(self, other: "VectorClock") -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        return self.happens_before(other)

    def __ge__(self, other: "VectorClock") -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        return other._le_pointwise(self)

    def __gt__(self, other: "VectorClock") -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        return other.happens_before(self)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.counters.items())))

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v}" for k, v in sorted(self.counters.items()))
        return f"VectorClock(node_id={self.node_id!r}, {items})"
