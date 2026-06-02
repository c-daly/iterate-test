"""Probabilistic skip list (Assignment 1).

A layered, sorted linked list giving O(log n) expected search / insert /
delete via randomized leveling. Keys are integers; values are arbitrary.
"""

from __future__ import annotations

import random
from typing import Any, Iterator


class _Node:
    """A skip-list node holding a key, value, and per-level forward pointers.

    ``forward[i]`` is the next node at level ``i`` (0-indexed). The header
    sentinel uses a key/value of ``None`` and exists only to anchor the
    forward pointers at every level.
    """

    __slots__ = ("key", "value", "forward")

    def __init__(self, key: Any, value: Any, level: int) -> None:
        self.key = key
        self.value = value
        # One forward pointer per level this node participates in.
        self.forward: list[_Node | None] = [None] * level


class SkipList:
    """A probabilistic skip list mapping integer keys to arbitrary values."""

    def __init__(self, max_level: int = 16, p: float = 0.5) -> None:
        if max_level < 1:
            raise ValueError("max_level must be >= 1")
        if not 0.0 < p < 1.0:
            raise ValueError("p must be in the open interval (0, 1)")

        self.max_level: int = max_level
        self.p: float = p
        # Current number of populated levels (>= 1). Grows toward max_level
        # as taller nodes are inserted.
        self.level: int = 1
        self._size: int = 0
        # Header sentinel carries forward pointers for the full max height so
        # the structure can grow without reallocating the header.
        self._header: _Node = _Node(None, None, max_level)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _random_level(self) -> int:
        """Draw a level from a geometric distribution with parameter ``p``.

        Returns a value in ``[1, max_level]``. Each additional level is taken
        with probability ``p`` (a coin flip), giving the geometric tail that
        yields O(log n) expected height.
        """
        level = 1
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level

    def _find_predecessors(self, key: int) -> list[_Node]:
        """Return, for each level, the last node whose key is < ``key``.

        ``update[i]`` is the node after which ``key`` would be inserted (or
        whose forward pointer at level ``i`` must be rewired on delete).
        """
        update: list[_Node] = [self._header] * self.max_level
        node = self._header
        for i in range(self.level - 1, -1, -1):
            nxt = node.forward[i]
            while nxt is not None and nxt.key < key:
                node = nxt
                nxt = node.forward[i]
            update[i] = node
        return update

    # ------------------------------------------------------------------ #
    # Core operations
    # ------------------------------------------------------------------ #
    def insert(self, key: int, value: Any) -> None:
        """Insert ``key`` -> ``value``; update in place if ``key`` exists."""
        update = self._find_predecessors(key)
        candidate = update[0].forward[0]
        if candidate is not None and candidate.key == key:
            # Key already present: update semantics, no structural change.
            candidate.value = value
            return

        node_level = self._random_level()
        if node_level > self.level:
            # New node is taller than the current list height. update[i] is
            # already self._header for i >= self.level (set by
            # _find_predecessors), so no explicit back-fill is needed.
            self.level = node_level

        node = _Node(key, value, node_level)
        for i in range(node_level):
            node.forward[i] = update[i].forward[i]
            update[i].forward[i] = node
        self._size += 1

    def delete(self, key: int) -> bool:
        """Remove ``key`` from every level. Return True iff it was present."""
        update = self._find_predecessors(key)
        target = update[0].forward[0]
        if target is None or target.key != key:
            return False

        # target participates only in levels 0..len(target.forward)-1, and
        # update[i] is its predecessor at each of those levels, so
        # update[i].forward[i] is guaranteed to be target there.
        for i in range(len(target.forward)):
            update[i].forward[i] = target.forward[i]
        # Shrink the height if the top levels are now empty.
        while self.level > 1 and self._header.forward[self.level - 1] is None:
            self.level -= 1
        self._size -= 1
        return True

    def search(self, key: int) -> Any | None:
        """Return the value bound to ``key``, or None if absent.

        .. note::
            ``None`` is returned for both a missing key *and* a key whose
            stored value is ``None``. Use ``key in sl`` (``__contains__``) to
            distinguish the two cases when ``None`` is a valid stored value.
        """
        node = self._header
        for i in range(self.level - 1, -1, -1):
            nxt = node.forward[i]
            while nxt is not None and nxt.key < key:
                node = nxt
                nxt = node.forward[i]
        candidate = node.forward[0]
        if candidate is not None and candidate.key == key:
            return candidate.value
        return None

    def range_query(self, lo: int, hi: int) -> list[tuple[int, Any]]:
        """Return sorted (key, value) pairs with ``lo <= key <= hi``.

        Inclusive on both bounds; an empty list when ``lo > hi`` or no key
        falls in range.
        """
        result: list[tuple[int, Any]] = []
        if lo > hi:
            return result
        # Descend to the first node with key >= lo, then walk level 0.
        node = self._header
        for i in range(self.level - 1, -1, -1):
            nxt = node.forward[i]
            while nxt is not None and nxt.key < lo:
                node = nxt
                nxt = node.forward[i]
        node = node.forward[0]
        while node is not None and node.key <= hi:
            result.append((node.key, node.value))
            node = node.forward[0]
        return result

    # ------------------------------------------------------------------ #
    # Dunder protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: int) -> bool:
        node = self._header
        for i in range(self.level - 1, -1, -1):
            nxt = node.forward[i]
            while nxt is not None and nxt.key < key:
                node = nxt
                nxt = node.forward[i]
        candidate = node.forward[0]
        return candidate is not None and candidate.key == key

    def __iter__(self) -> Iterator[tuple[int, Any]]:
        node = self._header.forward[0]
        while node is not None:
            yield (node.key, node.value)
            node = node.forward[0]

    def __repr__(self) -> str:
        items = ", ".join(f"{k}: {v!r}" for k, v in self)
        return f"SkipList({{{items}}})"
