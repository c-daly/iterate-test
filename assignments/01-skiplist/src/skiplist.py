"""Probabilistic skip list implementation."""

import random
from typing import Any, Iterator, List, Optional, Tuple


class _Node:
    """A node in the skip list with forward pointers for each level."""

    __slots__ = ('key', 'value', 'forward')

    def __init__(self, key: Any, value: Any, level: int) -> None:
        self.key = key
        self.value = value
        self.forward: List[Optional['_Node']] = [None] * (level + 1)


class SkipList:
    """A probabilistic skip list supporting insert, delete, search, and range queries.

    Keys must be comparable (support < and ==).
    """

    def __init__(self, max_level: int = 16, p: float = 0.5) -> None:
        self._max_level = max_level
        self._p = p
        self._level = 0  # current highest level in use
        self._size = 0
        self._header = _Node(None, None, max_level)

    def _random_level(self) -> int:
        """Generate a random level for a new node."""
        lvl = 0
        while random.random() < self._p and lvl < self._max_level:
            lvl += 1
        return lvl

    def insert(self, key: Any, value: Any) -> None:
        """Insert a key-value pair. If key exists, update its value."""
        # Build update array: update[i] is the last node at level i
        # whose key < the target key.
        update: List[Optional[_Node]] = [None] * (self._max_level + 1)
        current = self._header

        for i in range(self._level, -1, -1):
            while current.forward[i] is not None and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        # current is now the greatest node with key < target (or header)
        current = current.forward[0]

        if current is not None and current.key == key:
            # Key exists — update value
            current.value = value
            return

        # New key — generate random level and splice in
        new_level = self._random_level()

        if new_level > self._level:
            # Point new higher levels from header
            for i in range(self._level + 1, new_level + 1):
                update[i] = self._header
            self._level = new_level

        new_node = _Node(key, value, new_level)

        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

        self._size += 1

    def delete(self, key: Any) -> bool:
        """Delete a key. Returns True if key was found and deleted, False otherwise."""
        update: List[Optional[_Node]] = [None] * (self._max_level + 1)
        current = self._header

        for i in range(self._level, -1, -1):
            while current.forward[i] is not None and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        target = current.forward[0]

        if target is None or target.key != key:
            return False

        # Unlink the target node at every level it appears
        for i in range(self._level + 1):
            if update[i].forward[i] is not target:
                break
            update[i].forward[i] = target.forward[i]

        # Reduce level if top levels are now empty
        while self._level > 0 and self._header.forward[self._level] is None:
            self._level -= 1

        self._size -= 1
        return True

    def search(self, key: Any) -> Optional[Any]:
        """Search for a key. Returns the associated value, or None if not found."""
        current = self._header

        for i in range(self._level, -1, -1):
            while current.forward[i] is not None and current.forward[i].key < key:
                current = current.forward[i]

        current = current.forward[0]

        if current is not None and current.key == key:
            return current.value
        return None

    def range_query(self, lo: Any, hi: Any) -> List[Tuple[Any, Any]]:
        """Return all (key, value) pairs where lo <= key <= hi, in sorted order."""
        result: List[Tuple[Any, Any]] = []

        # Find the predecessor of lo
        current = self._header
        for i in range(self._level, -1, -1):
            while current.forward[i] is not None and current.forward[i].key < lo:
                current = current.forward[i]

        # Walk level 0, collecting entries while key <= hi
        current = current.forward[0]
        while current is not None and current.key <= hi:
            result.append((current.key, current.value))
            current = current.forward[0]

        return result

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: Any) -> bool:
        return self.search(key) is not None

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        """Iterate over all (key, value) pairs in sorted order."""
        node = self._header.forward[0]
        while node is not None:
            yield (node.key, node.value)
            node = node.forward[0]

    def __repr__(self) -> str:
        items = []
        node = self._header.forward[0]
        while node is not None:
            items.append(f"{node.key!r}: {node.value!r}")
            node = node.forward[0]
        return f"SkipList({{{', '.join(items)}}}, level={self._level}, size={self._size})"
