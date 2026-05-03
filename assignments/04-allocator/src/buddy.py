"""Buddy memory allocator.

Pool size must be a power of two. Requests are rounded up to the next
power of two; oversized free blocks are split, and freed blocks are
recursively coalesced with their buddy (offset XOR block_size).
"""
from __future__ import annotations

from .allocator import AllocationError, Allocator, AllocatorStats

_MIN_BLOCK = 1


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= ``n`` (assumes ``n >= 1``)."""
    p = 1
    while p < n:
        p <<= 1
    return p


class BuddyAllocator(Allocator):
    """Power-of-two buddy allocator with recursive coalescing on free."""

    def __init__(self, pool_size: int) -> None:
        if not _is_power_of_two(pool_size):
            raise ValueError("pool_size must be a power of two")
        super().__init__(pool_size)
        # Free lists keyed by block size (power of two). Each set holds offsets.
        # Sets give O(1) buddy lookup and removal during coalescing.
        self._free: dict[int, set[int]] = {pool_size: {0}}
        # offset -> size for live allocations.
        self._live: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def alloc(self, size: int) -> int:
        if size <= 0:
            raise AllocationError("size must be positive")
        block = max(_MIN_BLOCK, _next_power_of_two(size))
        if block > self.pool_size:
            raise AllocationError(f"requested {size} exceeds pool size {self.pool_size}")

        # Find the smallest available block >= requested.
        candidate = block
        while candidate <= self.pool_size and not self._free.get(candidate):
            candidate <<= 1
        if candidate > self.pool_size:
            raise AllocationError(f"no free block of size {block} available")

        offset = self._free[candidate].pop()
        # Split down to the requested block size.
        while candidate > block:
            candidate >>= 1
            buddy = offset + candidate
            self._free.setdefault(candidate, set()).add(buddy)
        self._live[offset] = block
        return offset

    def free(self, offset: int) -> None:
        if offset not in self._live:
            raise AllocationError(f"invalid free at offset {offset}")
        size = self._live.pop(offset)
        self._coalesce(offset, size)

    def stats(self) -> AllocatorStats:
        allocated = sum(self._live.values())
        free_blocks = [(off, sz) for sz, offsets in self._free.items() for off in offsets]
        free_total = sum(sz for _, sz in free_blocks)
        num_free = len(free_blocks)
        if free_total > 0 and num_free > 0:
            largest = max(sz for _, sz in free_blocks)
            fragmentation = 1.0 - (largest / free_total)
        else:
            fragmentation = 0.0
        return AllocatorStats(
            total_size=self.pool_size,
            allocated=allocated,
            free=free_total,
            num_allocations=len(self._live),
            num_free_blocks=num_free,
            fragmentation=fragmentation,
        )

    def dump(self) -> str:
        """Return a multi-line description of live and free blocks, sorted by offset."""
        live = [(off, sz, "USED") for off, sz in self._live.items()]
        free = [(off, sz, "FREE") for sz, offsets in self._free.items() for off in offsets]
        rows = sorted(live + free, key=lambda r: r[0])
        lines = [f"BuddyAllocator pool_size={self.pool_size}"]
        for off, sz, tag in rows:
            lines.append(f"  [{off:>6}..{off + sz - 1:<6}] size={sz:<6} {tag}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _coalesce(self, offset: int, size: int) -> None:
        """Recursively merge with buddy if the buddy is free and same-sized."""
        while size < self.pool_size:
            buddy = offset ^ size
            siblings = self._free.get(size)
            if siblings is not None and buddy in siblings:
                siblings.discard(buddy)
                offset = min(offset, buddy)
                size <<= 1
            else:
                break
        self._free.setdefault(size, set()).add(offset)
