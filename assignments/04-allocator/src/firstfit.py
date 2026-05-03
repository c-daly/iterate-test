"""First-fit memory allocator.

Maintains a sorted free list (by offset). On ``alloc`` the first block large
enough is used; if the remainder is at least :data:`_SPLIT_THRESHOLD` bytes
the block is split, otherwise the entire block is consumed. On ``free`` the
released range is coalesced with adjacent (left and right) free neighbours.
"""
from __future__ import annotations

from bisect import insort

from .allocator import AllocationError, Allocator, AllocatorStats

_SPLIT_THRESHOLD = 16


class FirstFitAllocator(Allocator):
    """First-fit allocator with adjacent-block coalescing."""

    def __init__(self, pool_size: int) -> None:
        super().__init__(pool_size)
        # Sorted list of (offset, size) tuples for free blocks.
        # Sorted by offset (the natural tuple ordering with unique offsets).
        self._free: list[tuple[int, int]] = [(0, pool_size)]
        # offset -> size for live allocations.
        self._live: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def alloc(self, size: int) -> int:
        if size <= 0:
            raise AllocationError("size must be positive")
        if size > self.pool_size:
            raise AllocationError(f"requested {size} exceeds pool size {self.pool_size}")

        for i, (off, blk) in enumerate(self._free):
            if blk >= size:
                remainder = blk - size
                # Pop the block; we will re-insert the remainder if any.
                self._free.pop(i)
                if remainder >= _SPLIT_THRESHOLD:
                    insort(self._free, (off + size, remainder))
                    self._live[off] = size
                else:
                    # Consume the whole block.
                    self._live[off] = blk
                return off
        raise AllocationError(f"no free block of size {size} available")

    def free(self, offset: int) -> None:
        if offset not in self._live:
            raise AllocationError(f"invalid free at offset {offset}")
        size = self._live.pop(offset)
        self._insert_and_coalesce(offset, size)

    def stats(self) -> AllocatorStats:
        allocated = sum(self._live.values())
        free_total = sum(sz for _, sz in self._free)
        num_free = len(self._free)
        if free_total > 0 and num_free > 0:
            largest = max(sz for _, sz in self._free)
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
        free = [(off, sz, "FREE") for off, sz in self._free]
        rows = sorted(live + free, key=lambda r: r[0])
        lines = [f"FirstFitAllocator pool_size={self.pool_size}"]
        for off, sz, tag in rows:
            lines.append(f"  [{off:>6}..{off + sz - 1:<6}] size={sz:<6} {tag}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _insert_and_coalesce(self, offset: int, size: int) -> None:
        """Insert a freed range and merge with any adjacent free neighbours."""
        insort(self._free, (offset, size))
        # Find our index after insertion (first match by offset).
        idx = next(i for i, (o, _) in enumerate(self._free) if o == offset)

        # Try to merge with right neighbour first.
        if idx + 1 < len(self._free):
            r_off, r_sz = self._free[idx + 1]
            if offset + size == r_off:
                self._free[idx] = (offset, size + r_sz)
                self._free.pop(idx + 1)
                size += r_sz

        # Then merge with left neighbour.
        if idx > 0:
            l_off, l_sz = self._free[idx - 1]
            if l_off + l_sz == offset:
                self._free[idx - 1] = (l_off, l_sz + size)
                self._free.pop(idx)
