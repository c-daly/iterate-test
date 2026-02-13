import math

from allocator import Allocator, AllocationError, AllocatorStats


class BuddyAllocator(Allocator):
    def __init__(self, pool_size: int):
        super().__init__(pool_size)
        if pool_size < 1 or (pool_size & (pool_size - 1)) != 0:
            raise ValueError("pool_size must be power of 2")
        self._min_block = 16
        max_order = int(math.log2(pool_size))
        min_order = int(math.log2(self._min_block))
        self._free_lists: dict[int, set[int]] = {
            i: set() for i in range(min_order, max_order + 1)
        }
        self._free_lists[max_order].add(0)
        self._allocs: dict[int, int] = {}  # offset -> order
        self._min_order = min_order
        self._max_order = max_order

    def alloc(self, size: int) -> int:
        if size <= 0:
            raise AllocationError(f"invalid allocation size: {size}")

        # Round up to next power of 2, minimum _min_block
        rounded = max(self._min_block, 1 << math.ceil(math.log2(max(size, 1))))
        target_order = int(math.log2(rounded))

        # Find smallest order >= target with a free block
        found_order = None
        for order in range(target_order, self._max_order + 1):
            if self._free_lists.get(order):
                found_order = order
                break

        if found_order is None:
            raise AllocationError(
                f"cannot allocate {size} bytes (rounded to {rounded}): no free block"
            )

        # Remove the block we found from its free list
        block = self._free_lists[found_order].pop()

        # Split down to target order, keeping the lower half and freeing the upper
        while found_order > target_order:
            found_order -= 1
            half_size = 1 << found_order
            # Put the upper buddy on the free list
            self._free_lists[found_order].add(block + half_size)
            # Keep splitting the lower half (block stays the same)

        # Record the allocation
        self._allocs[block] = target_order
        return block

    def free(self, offset: int) -> None:
        if offset not in self._allocs:
            raise AllocationError(f"invalid free: offset {offset} not allocated")

        order = self._allocs.pop(offset)

        # Coalesce with buddy
        while order < self._max_order:
            buddy = offset ^ (1 << order)
            if buddy in self._free_lists[order]:
                self._free_lists[order].remove(buddy)
                offset = min(offset, buddy)
                order += 1
            else:
                break

        self._free_lists[order].add(offset)

    def stats(self) -> AllocatorStats:
        allocated = sum(1 << order for order in self._allocs.values())
        free = self.pool_size - allocated
        num_free_blocks = sum(len(s) for s in self._free_lists.values())

        largest = max(
            (1 << order for order, s in self._free_lists.items() if s),
            default=0,
        )
        fragmentation = (1.0 - largest / free) if free > 0 else 0.0

        return AllocatorStats(
            total_size=self.pool_size,
            allocated=allocated,
            free=free,
            num_allocations=len(self._allocs),
            num_free_blocks=num_free_blocks,
            fragmentation=fragmentation,
        )
