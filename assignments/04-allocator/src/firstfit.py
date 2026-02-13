import bisect

from allocator import Allocator, AllocationError, AllocatorStats


class FirstFitAllocator(Allocator):
    def __init__(self, pool_size: int):
        super().__init__(pool_size)
        self._free_list = [(0, pool_size)]  # sorted by offset: [(offset, size), ...]
        self._allocs = {}  # offset -> size

    def alloc(self, size: int) -> int:
        if size <= 0:
            raise AllocationError("size must be positive")
        for i, (offset, block_size) in enumerate(self._free_list):
            if block_size >= size:
                self._free_list.pop(i)
                remainder = block_size - size
                if remainder >= 16:
                    # Insert remainder back, maintaining sorted order
                    self._insert_free(offset + size, remainder)
                self._allocs[offset] = size
                return offset
        raise AllocationError("no suitable block found")

    def free(self, offset: int) -> None:
        if offset not in self._allocs:
            raise AllocationError(f"invalid free: {offset}")
        size = self._allocs.pop(offset)
        self._insert_free(offset, size)
        self._coalesce()

    def _insert_free(self, offset, size):
        offsets = [o for o, _ in self._free_list]
        idx = bisect.bisect_left(offsets, offset)
        self._free_list.insert(idx, (offset, size))

    def _coalesce(self):
        i = 0
        while i < len(self._free_list) - 1:
            o1, s1 = self._free_list[i]
            o2, s2 = self._free_list[i + 1]
            if o1 + s1 == o2:
                self._free_list[i] = (o1, s1 + s2)
                self._free_list.pop(i + 1)
            else:
                i += 1

    def stats(self) -> AllocatorStats:
        allocated = sum(self._allocs.values())
        free = self.pool_size - allocated
        num_free_blocks = len(self._free_list)
        largest = max((s for _, s in self._free_list), default=0)
        frag = 1 - largest / free if free > 0 else 0.0
        return AllocatorStats(
            total_size=self.pool_size,
            allocated=allocated,
            free=free,
            num_allocations=len(self._allocs),
            num_free_blocks=num_free_blocks,
            fragmentation=frag,
        )
