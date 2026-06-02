"""Buddy memory allocator over a simulated pool.

Block sizes are powers of two between ``MIN_BLOCK`` and the pool size.
A request is rounded up to the nearest power-of-two block; a larger free
block is split down to size as needed. On free, a block is coalesced
with its buddy (``buddy = offset XOR block_size``) recursively as long
as the buddy is also free and of the same size.
"""

from allocator import AllocationError, Allocator, AllocatorStats

MIN_BLOCK = 16


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _round_up_pow2(n: int) -> int:
    """Smallest power of two >= ``n`` (and >= MIN_BLOCK)."""
    size = MIN_BLOCK
    while size < n:
        size <<= 1
    return size


class BuddyAllocator(Allocator):
    """Classic buddy-system allocator."""

    def __init__(self, pool_size: int):
        if not _is_power_of_two(pool_size):
            raise ValueError("pool_size must be a power of 2 for the buddy allocator")
        if pool_size < MIN_BLOCK:
            raise ValueError(f"pool_size must be >= {MIN_BLOCK}")
        super().__init__(pool_size)
        # free_lists[size] = set of free block offsets of that size.
        self._free: dict[int, set[int]] = {}
        size = MIN_BLOCK
        while size <= pool_size:
            self._free[size] = set()
            size <<= 1
        self._free[pool_size].add(0)
        # offset -> block_size for live allocations.
        self._live: dict[int, int] = {}

    # -- allocation -------------------------------------------------------

    def alloc(self, size: int) -> int:
        if size <= 0:
            raise AllocationError("allocation size must be positive")
        block = _round_up_pow2(size)
        if block > self.pool_size:
            raise AllocationError(
                f"request of {size} (block {block}) exceeds pool {self.pool_size}"
            )
        # Find the smallest available block >= the requested block size.
        avail = block
        while avail <= self.pool_size and not self._free[avail]:
            avail <<= 1
        if avail > self.pool_size:
            raise AllocationError("out of memory")
        # Split down to the target block size.
        offset = self._free[avail].pop()
        while avail > block:
            avail >>= 1
            buddy = offset ^ avail  # the upper half becomes a free buddy
            self._free[avail].add(buddy)
        self._live[offset] = block
        return offset

    # -- freeing ----------------------------------------------------------

    def free(self, offset: int) -> None:
        if offset not in self._live:
            raise AllocationError(f"invalid or already-freed offset: {offset}")
        block = self._live.pop(offset)
        # Coalesce recursively with the buddy while it is free + same size.
        size = block
        while size < self.pool_size:
            buddy = offset ^ size
            if buddy in self._free[size]:
                self._free[size].discard(buddy)
                offset = min(offset, buddy)
                size <<= 1
            else:
                break
        self._free[size].add(offset)

    # -- introspection ----------------------------------------------------

    def _free_block_offsets(self) -> list[tuple[int, int]]:
        """Return (offset, size) for every free block, sorted by offset."""
        blocks = []
        for size, offs in self._free.items():
            for off in offs:
                blocks.append((off, size))
        blocks.sort()
        return blocks

    def stats(self) -> AllocatorStats:
        allocated = sum(self._live.values())
        free = self.pool_size - allocated
        num_free_blocks = sum(len(offs) for offs in self._free.values())
        if free > 0 and num_free_blocks > 0:
            largest = max(
                (size for size, offs in self._free.items() if offs), default=0
            )
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
        parts = [f"BuddyAllocator(pool={self.pool_size})"]
        for off, size in sorted(self._live.items()):
            parts.append(f"  [ALLOC] off={off:>6} size={size}")
        for off, size in self._free_block_offsets():
            parts.append(f"  [FREE ] off={off:>6} size={size}")
        return "\n".join(parts)
