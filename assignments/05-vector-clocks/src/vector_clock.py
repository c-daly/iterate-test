from __future__ import annotations


class VectorClock:
    """Vector clock for tracking causality in distributed systems."""

    def __init__(self, node_id: str, num_nodes: int) -> None:
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.clock: dict[str, int] = {node_id: 0}

    def increment(self) -> VectorClock:
        """Increment this node's counter. Returns a new VectorClock."""
        new = VectorClock.__new__(VectorClock)
        new.node_id = self.node_id
        new.num_nodes = self.num_nodes
        new.clock = dict(self.clock)
        new.clock[self.node_id] = new.clock.get(self.node_id, 0) + 1
        return new

    def merge(self, other: VectorClock) -> VectorClock:
        """Merge with another clock (element-wise max), then increment local."""
        new = VectorClock.__new__(VectorClock)
        new.node_id = self.node_id
        new.num_nodes = self.num_nodes
        all_keys = set(self.clock) | set(other.clock)
        new.clock = {}
        for k in all_keys:
            new.clock[k] = max(self.clock.get(k, 0), other.clock.get(k, 0))
        new.clock[self.node_id] = new.clock.get(self.node_id, 0) + 1
        return new

    def happens_before(self, other: VectorClock) -> bool:
        """True if self strictly happens-before other (irreflexive)."""
        # self <= other and self != other
        # All entries in self must be <= corresponding in other
        # and at least one must be strictly less.
        all_keys = set(self.clock) | set(other.clock)
        at_least_one_less = False
        for k in all_keys:
            s = self.clock.get(k, 0)
            o = other.clock.get(k, 0)
            if s > o:
                return False
            if s < o:
                at_least_one_less = True
        return at_least_one_less

    def is_concurrent(self, other: VectorClock) -> bool:
        """True if neither happens-before the other and they are not equal."""
        return (
            not self.happens_before(other)
            and not other.happens_before(self)
            and self != other
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        all_keys = set(self.clock) | set(other.clock)
        return all(
            self.clock.get(k, 0) == other.clock.get(k, 0) for k in all_keys
        )

    def __le__(self, other: VectorClock) -> bool:
        all_keys = set(self.clock) | set(other.clock)
        return all(
            self.clock.get(k, 0) <= other.clock.get(k, 0) for k in all_keys
        )

    def __lt__(self, other: VectorClock) -> bool:
        return self <= other and self != other

    def __repr__(self) -> str:
        return f"VectorClock({self.node_id}, {self.clock})"

    def __hash__(self) -> int:
        return hash((self.node_id, tuple(sorted(self.clock.items()))))
