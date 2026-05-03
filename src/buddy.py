"""Buddy allocator with power-of-2 block sizes."""
from __future__ import annotations

from .allocator import (
    AllocationError,
    Allocator,
    AllocatorStats,
    compute_fragmentation,
)


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


class BuddyAllocator(Allocator):
    """Classic buddy allocator.

    - Pool size must be a power of two.
    - Requests are rounded up to the next power of two.
    - Free blocks of equal size whose buddy address (offset XOR size)
      is also free are coalesced recursively.
    """

    MIN_BLOCK = 1  # smallest split unit

    def __init__(self, pool_size: int) -> None:
        super().__init__(pool_size)
        if not _is_power_of_two(pool_size):
            raise ValueError("BuddyAllocator pool_size must be a power of two")
        # free_lists[size] -> set of offsets of free blocks of that size
        self._free_lists: dict[int, set[int]] = {pool_size: {0}}
        # offset -> block_size of currently allocated blocks
        self._allocated: dict[int, int] = {}

    # ---- internal helpers ----
    def _smallest_size_for(self, size: int) -> int:
        return max(self.MIN_BLOCK, _next_pow2(size))

    def _split_until(self, target_size: int) -> int:
        """Find a free block >= target_size and split down to target_size.

        Returns the offset of a free block of exactly target_size.
        Raises AllocationError if no such block exists.
        """
        # Find the smallest available size >= target_size
        size = target_size
        while size <= self.pool_size:
            if self._free_lists.get(size):
                break
            size <<= 1
        else:
            raise AllocationError("no block large enough")

        # Pop a free block of `size`
        offset = next(iter(self._free_lists[size]))
        self._free_lists[size].remove(offset)
        if not self._free_lists[size]:
            del self._free_lists[size]

        # Split down
        while size > target_size:
            size >>= 1
            buddy = offset + size
            self._free_lists.setdefault(size, set()).add(buddy)
        return offset

    # ---- public API ----
    def alloc(self, size: int) -> int:
        if size <= 0:
            raise AllocationError("size must be positive")
        block_size = self._smallest_size_for(size)
        if block_size > self.pool_size:
            raise AllocationError("request exceeds pool size")
        offset = self._split_until(block_size)
        self._allocated[offset] = block_size
        return offset

    def free(self, offset: int) -> None:
        if offset not in self._allocated:
            raise AllocationError(f"invalid free at offset {offset}")
        size = self._allocated.pop(offset)
        # Coalesce recursively
        while size < self.pool_size:
            buddy = offset ^ size
            buddies = self._free_lists.get(size)
            if buddies and buddy in buddies:
                buddies.remove(buddy)
                if not buddies:
                    del self._free_lists[size]
                offset = min(offset, buddy)
                size <<= 1
            else:
                break
        self._free_lists.setdefault(size, set()).add(offset)

    def stats(self) -> AllocatorStats:
        allocated = sum(self._allocated.values())
        free_blocks: list[int] = []
        num_free_blocks = 0
        for sz, offsets in self._free_lists.items():
            for _ in offsets:
                free_blocks.append(sz)
                num_free_blocks += 1
        total_free = self.pool_size - allocated
        return AllocatorStats(
            total_size=self.pool_size,
            allocated=allocated,
            free=total_free,
            num_allocations=len(self._allocated),
            num_free_blocks=num_free_blocks,
            fragmentation=compute_fragmentation(free_blocks, total_free),
        )

    def dump(self) -> str:
        lines = [super().dump()]
        for sz in sorted(self._free_lists):
            offs = sorted(self._free_lists[sz])
            lines.append(f"  free size={sz}: {offs}")
        for off in sorted(self._allocated):
            lines.append(f"  alloc offset={off} size={self._allocated[off]}")
        return "\n".join(lines)
