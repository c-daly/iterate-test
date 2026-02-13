class VectorClock:
    def __init__(self, node_id: str, num_nodes: int):
        self.node_id = node_id
        self.clock = {}  # node_id -> counter

    def increment(self) -> 'VectorClock':
        self.clock[self.node_id] = self.clock.get(self.node_id, 0) + 1
        return self

    def merge(self, other: 'VectorClock') -> 'VectorClock':
        for k in set(self.clock) | set(other.clock):
            self.clock[k] = max(self.clock.get(k, 0), other.clock.get(k, 0))
        self.clock[self.node_id] = self.clock.get(self.node_id, 0) + 1
        return self

    def happens_before(self, other: 'VectorClock') -> bool:
        all_keys = set(self.clock) | set(other.clock)
        all_leq = all(self.clock.get(k, 0) <= other.clock.get(k, 0) for k in all_keys)
        any_lt = any(self.clock.get(k, 0) < other.clock.get(k, 0) for k in all_keys)
        return all_leq and any_lt

    def is_concurrent(self, other: 'VectorClock') -> bool:
        return not self.happens_before(other) and not other.happens_before(self) and self != other

    def __eq__(self, other):
        if not isinstance(other, VectorClock):
            return NotImplemented
        all_keys = set(self.clock) | set(other.clock)
        return all(self.clock.get(k, 0) == other.clock.get(k, 0) for k in all_keys)

    def __le__(self, other):
        all_keys = set(self.clock) | set(other.clock)
        return all(self.clock.get(k, 0) <= other.clock.get(k, 0) for k in all_keys)

    def __lt__(self, other):
        return self.happens_before(other)

    def __repr__(self):
        return f'VectorClock({self.node_id}, {dict(self.clock)})'

    def copy(self) -> 'VectorClock':
        vc = VectorClock(self.node_id, 0)
        vc.clock = dict(self.clock)
        return vc
