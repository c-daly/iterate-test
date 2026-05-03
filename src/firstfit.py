"""First-fit allocator with sorted free list and adjacent-block coalescing."""
from __future__ import annotations

from .allocator import (
    AllocationError,
    Allocator,
    AllocatorStats,
    compute_fragmentation,
)

MIN_SPLIT_REMAINDER = 16


class FirstFitAllocator(Allocator):
    """First-fit allocator over a single contiguous pool.

    Free list is kept sorted by offset. Adjacent free blocks are
    coalesced on free(). Splits only when the leftover would be
    >= MIN_SPLIT_REMAINDER bytes.
    """

    def __init__(self, pool_size: int) -> None:
        super().__init__(pool_size)
        # Sorted list of (offset, size) free blocks.
        self._free: list[list[int]] = [[0, pool_size]]
        # offset -> size of allocated blocks.
        self._allocated: dict[int, int] = {}

    def alloc(self, size: int) -> int:
        if size <= 0:
            raise AllocationError("size must be positive")
        for i, (off, blk) in enumerate(self._free):
            if blk >= size:
                remainder = blk - size
                if remainder >= MIN_SPLIT_REMAINDER:
                    # Split: shrink free entry from the left.
                    self._free[i] = [off + size, remainder]
                else:
                    # Consume the whole free block.
                    size = blk
                    self._free.pop(i)
                self._allocated[off] = size
                return off
        raise AllocationError("no free block large enough")

    def free(self, offset: int) -> None:
        if offset not in self._allocated:
            raise AllocationError(f"invalid free at offset {offset}")
        size = self._allocated.pop(offset)

        # Insert into sorted free list, then coalesce with neighbours.
        i = 0
        while i < len(self._free) and self._free[i][0] < offset:
            i += 1
        self._free.insert(i, [offset, size])

        # Coalesce with right neighbour.
        if i + 1 < len(self._free):
            r_off, r_size = self._free[i + 1]
            if offset + size == r_off:
                self._free[i][1] += r_size
                self._free.pop(i + 1)
        # Coalesce with left neighbour.
        if i > 0:
            l_off, l_size = self._free[i - 1]
            if l_off + l_size == self._free[i][0]:
                self._free[i - 1][1] += self._free[i][1]
                self._free.pop(i)

    def stats(self) -> AllocatorStats:
        allocated = sum(self._allocated.values())
        free_sizes = [sz for _, sz in self._free]
        total_free = self.pool_size - allocated
        return AllocatorStats(
            total_size=self.pool_size,
            allocated=allocated,
            free=total_free,
            num_allocations=len(self._allocated),
            num_free_blocks=len(self._free),
            fragmentation=compute_fragmentation(free_sizes, total_free),
        )

    def dump(self) -> str:
        lines = [super().dump()]
        for off, sz in self._free:
            lines.append(f"  free [{off}..{off + sz}) size={sz}")
        for off in sorted(self._allocated):
            sz = self._allocated[off]
            lines.append(f"  alloc [{off}..{off + sz}) size={sz}")
        return "\n".join(lines)
