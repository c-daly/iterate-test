"""Probabilistic Skip List implementation."""

from __future__ import annotations

import random
from typing import Any, Iterator


class _Node:
    """Internal node for the skip list."""

    __slots__ = ("key", "value", "forward")

    def __init__(self, key: int | None, value: Any, level: int):
        self.key = key
        self.value = value
        # forward[i] is the next node at level i
        self.forward: list[_Node | None] = [None] * (level + 1)


class SkipList:
    """A probabilistic skip list mapping integer keys to arbitrary values.

    Provides O(log n) expected time for search, insert, and delete
    using randomized leveling with geometric distribution.
    """

    def __init__(self, max_level: int = 16, p: float = 0.5):
        self._max_level = max_level
        self._p = p
        self._level = 0  # current highest level in use
        # Sentinel header node — key is None, never matches a real key
        self._header = _Node(None, None, max_level)
        self._size = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _random_level(self) -> int:
        """Generate a random level using geometric distribution with param p."""
        lvl = 0
        while random.random() < self._p and lvl < self._max_level:
            lvl += 1
        return lvl

    def _find_update(self, key: int) -> list[_Node]:
        """Return the update vector: update[i] is the last node at level i
        whose key is < the given key."""
        update: list[_Node] = [self._header] * (self._max_level + 1)
        current = self._header
        for i in range(self._level, -1, -1):
            while (
                current.forward[i] is not None
                and current.forward[i].key is not None
                and current.forward[i].key < key
            ):
                current = current.forward[i]
            update[i] = current
        return update

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, key: int, value: Any) -> None:
        """Insert a key-value pair. If key exists, update its value."""
        update = self._find_update(key)
        candidate = update[0].forward[0]

        if candidate is not None and candidate.key == key:
            # Key already exists — update value
            candidate.value = value
            return

        new_level = self._random_level()

        # If new level is higher than current, point header at the new node
        if new_level > self._level:
            for i in range(self._level + 1, new_level + 1):
                update[i] = self._header
            self._level = new_level

        new_node = _Node(key, value, new_level)
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

        self._size += 1

    def delete(self, key: int) -> bool:
        """Remove key from all levels. Return True if found, False otherwise."""
        update = self._find_update(key)
        candidate = update[0].forward[0]

        if candidate is None or candidate.key != key:
            return False

        for i in range(self._level + 1):
            if update[i].forward[i] is not candidate:
                break
            update[i].forward[i] = candidate.forward[i]

        # Shrink level if top levels are now empty
        while self._level > 0 and self._header.forward[self._level] is None:
            self._level -= 1

        self._size -= 1
        return True

    def search(self, key: int) -> Any | None:
        """Return the value associated with key, or None if not found."""
        current = self._header
        for i in range(self._level, -1, -1):
            while (
                current.forward[i] is not None
                and current.forward[i].key is not None
                and current.forward[i].key < key
            ):
                current = current.forward[i]
        current = current.forward[0]
        if current is not None and current.key == key:
            return current.value
        return None

    def range_query(self, lo: int, hi: int) -> list[tuple[int, Any]]:
        """Return sorted list of (key, value) tuples where lo <= key <= hi."""
        if lo > hi:
            return []

        result: list[tuple[int, Any]] = []
        # Navigate to the first node with key >= lo
        current = self._header
        for i in range(self._level, -1, -1):
            while (
                current.forward[i] is not None
                and current.forward[i].key is not None
                and current.forward[i].key < lo
            ):
                current = current.forward[i]
        current = current.forward[0]

        while current is not None and current.key is not None and current.key <= hi:
            result.append((current.key, current.value))
            current = current.forward[0]

        return result

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: int) -> bool:
        return self.search(key) is not None

    def __iter__(self) -> Iterator[tuple[int, Any]]:
        node = self._header.forward[0]
        while node is not None:
            yield (node.key, node.value)
            node = node.forward[0]

    def __repr__(self) -> str:
        items = ", ".join(f"{k}: {v!r}" for k, v in self)
        return f"SkipList({{{items}}}, level={self._level}, size={self._size})"
