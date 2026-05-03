"""Probabilistic skip list with O(log n) expected search/insert/delete.

A skip list maintains a sorted set of (key, value) pairs across multiple
layers of singly-linked forward pointers. Each node is promoted to a
randomly chosen level via independent coin flips with probability p.
This implementation follows William Pughs 1990 design and exposes a small
dictionary-like API.
"""

from __future__ import annotations

import random
from typing import Any, Iterator, Optional

__all__ = ["SkipList"]


class _Node:
    __slots__ = ("key", "value", "forward")

    def __init__(self, key: int, value: Any, level: int) -> None:
        self.key: int = key
        self.value: Any = value
        self.forward: list[Optional[_Node]] = [None] * level


class SkipList:
    """Probabilistic skip list keyed by integers."""

    def __init__(self, max_level: int = 16, p: float = 0.5) -> None:
        if max_level < 1:
            raise ValueError("max_level must be >= 1")
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0.0, 1.0]")
        self.max_level: int = max_level
        self.p: float = p
        self._header: _Node = _Node(key=-1, value=None, level=max_level)
        self._level: int = 1
        self._size: int = 0
        self._rng: random.Random = random.Random()

    def _random_level(self) -> int:
        level = 1
        while level < self.max_level and self._rng.random() < self.p:
            level += 1
        return level

    def _find_predecessors(self, key: int) -> list[_Node]:
        update: list[_Node] = [self._header] * self.max_level
        node = self._header
        for i in range(self._level - 1, -1, -1):
            nxt = node.forward[i]
            while nxt is not None and nxt.key < key:
                node = nxt
                nxt = node.forward[i]
            update[i] = node
        return update

    def insert(self, key: int, value: Any) -> None:
        update = self._find_predecessors(key)
        candidate = update[0].forward[0]
        if candidate is not None and candidate.key == key:
            candidate.value = value
            return
        new_level = self._random_level()
        if new_level > self._level:
            for i in range(self._level, new_level):
                update[i] = self._header
            self._level = new_level
        node = _Node(key=key, value=value, level=new_level)
        for i in range(new_level):
            node.forward[i] = update[i].forward[i]
            update[i].forward[i] = node
        self._size += 1

    def delete(self, key: int) -> bool:
        update = self._find_predecessors(key)
        target = update[0].forward[0]
        if target is None or target.key != key:
            return False
        for i in range(self._level):
            if update[i].forward[i] is not target:
                break
            update[i].forward[i] = target.forward[i]
        while self._level > 1 and self._header.forward[self._level - 1] is None:
            self._level -= 1
        self._size -= 1
        return True

    def search(self, key: int) -> Any | None:
        node = self._header
        for i in range(self._level - 1, -1, -1):
            nxt = node.forward[i]
            while nxt is not None and nxt.key < key:
                node = nxt
                nxt = node.forward[i]
        candidate = node.forward[0]
        if candidate is not None and candidate.key == key:
            return candidate.value
        return None

    def range_query(self, lo: int, hi: int) -> list[tuple[int, Any]]:
        if lo > hi:
            return []
        update = self._find_predecessors(lo)
        result: list[tuple[int, Any]] = []
        node = update[0].forward[0]
        while node is not None and node.key <= hi:
            result.append((node.key, node.value))
            node = node.forward[0]
        return result

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, int):
            return False
        return self.search(key) is not None

    def __iter__(self) -> Iterator[tuple[int, Any]]:
        node = self._header.forward[0]
        while node is not None:
            yield node.key, node.value
            node = node.forward[0]

    def __repr__(self) -> str:
        return (
            f"SkipList(size={self._size}, level={self._level}, "
            f"max_level={self.max_level}, p={self.p})"
        )
