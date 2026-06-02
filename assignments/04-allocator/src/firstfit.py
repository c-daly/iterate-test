"""First-fit memory allocator over a simulated pool.

Maintains a free list of ``(offset, size)`` regions kept sorted by
offset. Allocation scans from the start and takes the first region that
fits. If the leftover after carving out the request is at least
``MIN_SPLIT`` bytes, the region is split and the remainder returned to
the free list; otherwise the whole region is handed out (internal
fragmentation). On free, the region is returned to the free list and
coalesced with any immediately adjacent free regions.
"""

import bisect

from allocator import AllocationError, Allocator, AllocatorStats

MIN_SPLIT = 16


class FirstFitAllocator(Allocator):
    """First-fit allocator with adjacent-block coalescing."""

    def __init__(self, pool_size: int):
        super().__init__(pool_size)
        # Free list of (offset, size), kept sorted by offset.
        self._free: list[list[int]] = [[0, pool_size]]
        # offset -> size for live allocations.
        self._live: dict[int, int] = {}

    # -- allocation -------------------------------------------------------

    def alloc(self, size: int) -> int:
        if size <= 0:
            raise AllocationError("allocation size must be positive")
        if size > self.pool_size:
            raise AllocationError(
                f"request of {size} exceeds pool {self.pool_size}"
            )
        for i, (off, blk) in enumerate(self._free):
            if blk < size:
                continue
            remainder = blk - size
            if remainder >= MIN_SPLIT:
                # Split: carve the request off the front, shrink the hole.
                self._free[i] = [off + size, remainder]
                self._live[off] = size
            else:
                # Remainder too small to be useful: hand out the whole block.
                self._free.pop(i)
                self._live[off] = blk
            return off
        raise AllocationError("out of memory")

    # -- freeing ----------------------------------------------------------

    def free(self, offset: int) -> None:
        if offset not in self._live:
            raise AllocationError(f"invalid or already-freed offset: {offset}")
        size = self._live.pop(offset)
        # Insert the freed region into the already-sorted free list in
        # O(log n) search + O(n) shift, avoiding a full re-sort.
        bisect.insort(self._free, [offset, size])
        self._coalesce()

    def _coalesce(self) -> None:
        """Merge any free regions that are physically adjacent."""
        merged: list[list[int]] = []
        for off, blk in self._free:
            if merged and merged[-1][0] + merged[-1][1] == off:
                merged[-1][1] += blk
            else:
                merged.append([off, blk])
        self._free = merged

    # -- introspection ----------------------------------------------------

    def stats(self) -> AllocatorStats:
        allocated = sum(self._live.values())
        free = self.pool_size - allocated
        num_free_blocks = len(self._free)
        if free > 0 and self._free:
            largest = max(blk for _, blk in self._free)
            fragmentation = 1.0 - (largest / free)
        else:
            fragmentation = 0.0
        return AllocatorStats(
            total_size=self.pool_size,
            allocated=allocated,
            free=free,
            num_allocations=len(self._live),
            num_free_blocks=num_free_blocks,
            fragmentation=fragmentation,
        )

    def dump(self) -> str:
        parts = [f"FirstFitAllocator(pool={self.pool_size})"]
        for off, size in sorted(self._live.items()):
            parts.append(f"  [ALLOC] off={off:>6} size={size}")
        for off, size in self._free:
            parts.append(f"  [FREE ] off={off:>6} size={size}")
        return "\n".join(parts)
