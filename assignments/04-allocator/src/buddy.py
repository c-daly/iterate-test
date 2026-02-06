"""Buddy allocator implementation."""

from __future__ import annotations

import math

from src.allocator import Allocator, AllocationError, AllocatorStats


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << math.ceil(math.log2(n))


# Minimum block size for buddy allocator.
_MIN_BLOCK = 16


class BuddyAllocator(Allocator):
    def __init__(self, pool_size: int) -> None:
        if not _is_power_of_2(pool_size):
            raise ValueError(f"Pool size must be power of 2, got {pool_size}")
        super().__init__(pool_size)

        # free_lists: maps block_size -> set of offsets
        self._free: dict[int, set[int]] = {}
        self._free[pool_size] = {0}

        # allocated: offset -> block_size
        self._allocated: dict[int, int] = {}

    # ---- public API -------------------------------------------------------

    def alloc(self, size: int) -> int:
        if size <= 0:
            raise AllocationError("Cannot allocate zero or negative size")

        needed = max(_MIN_BLOCK, _next_power_of_2(size))
        if needed > self.pool_size:
            raise AllocationError(
                f"Requested {size} (rounded to {needed}) exceeds pool {self.pool_size}"
            )

        # Find the smallest available block >= needed
        block_size = needed
        while block_size <= self.pool_size:
            if block_size in self._free and self._free[block_size]:
                break
            block_size <<= 1
        else:
            raise AllocationError(f"No block available for size {size}")

        # Pop one free block of that size
        offset = min(self._free[block_size])  # deterministic: pick lowest
        self._free[block_size].remove(offset)
        if not self._free[block_size]:
            del self._free[block_size]

        # Split down to needed size
        while block_size > needed:
            block_size >>= 1
            buddy_offset = offset + block_size
            self._free.setdefault(block_size, set()).add(buddy_offset)

        self._allocated[offset] = needed
        return offset

    def free(self, offset: int) -> None:
        if offset not in self._allocated:
            raise AllocationError(f"Invalid free: offset {offset} not allocated")

        block_size = self._allocated.pop(offset)

        # Coalesce with buddy recursively
        while block_size < self.pool_size:
            buddy = offset ^ block_size
            if block_size in self._free and buddy in self._free[block_size]:
                # Buddy is free -- coalesce
                self._free[block_size].remove(buddy)
                if not self._free[block_size]:
                    del self._free[block_size]
                # Merged block starts at lower address
                offset = min(offset, buddy)
                block_size <<= 1
            else:
                break

        self._free.setdefault(block_size, set()).add(offset)

    def stats(self) -> AllocatorStats:
        allocated = sum(self._allocated.values())
        free = self.pool_size - allocated
        num_allocs = len(self._allocated)
        num_free = sum(len(s) for s in self._free.values())

        # External fragmentation: 1 - (largest_free_block / total_free)
        if free == 0:
            frag = 0.0
        else:
            largest = max(
                (sz for sz, offsets in self._free.items() if offsets),
                default=0,
            )
            frag = 1.0 - (largest / free) if free > 0 else 0.0

        return AllocatorStats(
            total_size=self.pool_size,
            allocated=allocated,
            free=free,
            num_allocations=num_allocs,
            num_free_blocks=num_free,
            fragmentation=frag,
        )

    def dump(self) -> str:
        lines: list[str] = [f"BuddyAllocator(pool={self.pool_size})"]
        lines.append("Allocated:")
        for offset in sorted(self._allocated):
            lines.append(f"  [{offset}:{offset + self._allocated[offset]}]")
        lines.append("Free:")
        for size in sorted(self._free):
            for offset in sorted(self._free[size]):
                lines.append(f"  [{offset}:{offset + size}] (size={size})")
        return "\n".join(lines)
