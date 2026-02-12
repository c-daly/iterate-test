"""Probabilistic skip list implementation."""

from __future__ import annotations

import random
from typing import Any, Iterator


class _Node:
    """A node in the skip list with forward pointers at multiple levels."""

    __slots__ = ("key", "value", "forward")

    def __init__(self, key: int | None, value: Any, level: int) -> None:
        self.key = key
        self.value = value
        self.forward: list[_Node | None] = [None] * (level + 1)


class SkipList:
    """Probabilistic skip list supporting insert, delete, search, and range queries.

    Args:
        max_level: Maximum number of levels (layers) in the skip list.
        p: Probability used for geometric level generation.
    """

    def __init__(self, max_level: int = 16, p: float = 0.5) -> None:
        self._max_level = max_level
        self._p = p
        self._level = 0  # current highest level in use
        self._header = _Node(None, None, max_level)
        self._size = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_level(self) -> int:
        """Generate a random level using geometric distribution with probability p."""
        lvl = 0
        while random.random() < self._p and lvl < self._max_level:
            lvl += 1
        return lvl

    def _find_update(self, key: int) -> list[_Node]:
        """Return the update vector: for each level, the last node before *key*."""
        update: list[_Node] = [self._header] * (self._max_level + 1)
        current = self._header
        for i in range(self._level, -1, -1):
            while current.forward[i] is not None and current.forward[i].key < key:  # type: ignore[operator]
                current = current.forward[i]  # type: ignore[assignment]
            update[i] = current
        return update

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, key: int, value: Any) -> None:
        """Insert or update a key-value pair."""
        update = self._find_update(key)
        candidate = update[0].forward[0]

        if candidate is not None and candidate.key == key:
            # Key already exists — update value
            candidate.value = value
            return

        new_level = self._random_level()

        # If new level exceeds current level, point extra update entries to header
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
        """Remove *key* from all levels. Return True if found, False otherwise."""
        update = self._find_update(key)
        candidate = update[0].forward[0]

        if candidate is None or candidate.key != key:
            return False

        for i in range(self._level + 1):
            if update[i].forward[i] is not candidate:
                break
            update[i].forward[i] = candidate.forward[i]

        # Shrink level if top layers are now empty
        while self._level > 0 and self._header.forward[self._level] is None:
            self._level -= 1

        self._size -= 1
        return True

    def search(self, key: int) -> Any | None:
        """Return the value associated with *key*, or None if not found."""
        current = self._header
        for i in range(self._level, -1, -1):
            while current.forward[i] is not None and current.forward[i].key < key:  # type: ignore[operator]
                current = current.forward[i]  # type: ignore[assignment]
        candidate = current.forward[0]
        if candidate is not None and candidate.key == key:
            return candidate.value
        return None

    def range_query(self, lo: int, hi: int) -> list[tuple[int, Any]]:
        """Return sorted (key, value) pairs where lo <= key <= hi."""
        result: list[tuple[int, Any]] = []
        current = self._header
        # Navigate to the first node with key >= lo
        for i in range(self._level, -1, -1):
            while current.forward[i] is not None and current.forward[i].key < lo:  # type: ignore[operator]
                current = current.forward[i]  # type: ignore[assignment]
        current = current.forward[0]
        while current is not None and current.key <= hi:  # type: ignore[operator]
            result.append((current.key, current.value))  # type: ignore[arg-type]
            current = current.forward[0]
        return result

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: int) -> bool:  # type: ignore[override]
        return self.search(key) is not None if not self._is_none_value(key) else self._key_exists(key)

    def _key_exists(self, key: int) -> bool:
        """Check if key exists (handles None values correctly)."""
        current = self._header
        for i in range(self._level, -1, -1):
            while current.forward[i] is not None and current.forward[i].key < key:  # type: ignore[operator]
                current = current.forward[i]  # type: ignore[assignment]
        candidate = current.forward[0]
        return candidate is not None and candidate.key == key

    def _is_none_value(self, key: int) -> bool:
        """Check if a key maps to None value (which makes search ambiguous)."""
        current = self._header
        for i in range(self._level, -1, -1):
            while current.forward[i] is not None and current.forward[i].key < key:  # type: ignore[operator]
                current = current.forward[i]  # type: ignore[assignment]
        candidate = current.forward[0]
        return candidate is not None and candidate.key == key and candidate.value is None

    def __iter__(self) -> Iterator[tuple[int, Any]]:
        current = self._header.forward[0]
        while current is not None:
            yield (current.key, current.value)  # type: ignore[misc]
            current = current.forward[0]

    def __repr__(self) -> str:
        return f"SkipList(len={self._size}, levels={self._level + 1})"
