"""First-fit allocator implementation."""

from __future__ import annotations

from src.allocator import Allocator, AllocationError, AllocatorStats

# Minimum remainder size to justify splitting a block.
_MIN_SPLIT = 16


class _Block:
    """A block in the memory pool (allocated or free)."""

    __slots__ = ("offset", "size", "is_free")

    def __init__(self, offset: int, size: int, *, is_free: bool = True) -> None:
        self.offset = offset
        self.size = size
        self.is_free = is_free

    def __repr__(self) -> str:
        state = "FREE" if self.is_free else "USED"
        return f"Block({self.offset}, {self.size}, {state})"


class FirstFitAllocator(Allocator):
    def __init__(self, pool_size: int) -> None:
        super().__init__(pool_size)
        # Maintain a list of blocks sorted by offset.
        self._blocks: list[_Block] = [_Block(0, pool_size, is_free=True)]

    # ---- helpers ----------------------------------------------------------

    def _find_block_index(self, offset: int) -> int | None:
        """Return index of allocated block at *offset*, or None."""
        for i, b in enumerate(self._blocks):
            if b.offset == offset and not b.is_free:
                return i
        return None

    def _coalesce(self) -> None:
        """Merge adjacent free blocks."""
        i = 0
        while i < len(self._blocks) - 1:
            if self._blocks[i].is_free and self._blocks[i + 1].is_free:
                self._blocks[i].size += self._blocks[i + 1].size
                del self._blocks[i + 1]
            else:
                i += 1

    # ---- public API -------------------------------------------------------

    def alloc(self, size: int) -> int:
        if size <= 0:
            raise AllocationError("Cannot allocate zero or negative size")

        for i, block in enumerate(self._blocks):
            if not block.is_free or block.size < size:
                continue

            # Found a suitable block -- first fit.
            remainder = block.size - size
            if remainder >= _MIN_SPLIT:
                # Split: shrink this block, insert new free block after it
                new_free = _Block(
                    block.offset + size, remainder, is_free=True,
                )
                block.size = size
                block.is_free = False
                self._blocks.insert(i + 1, new_free)
            else:
                # Use entire block (no split)
                block.is_free = False

            return block.offset

        raise AllocationError(f"No suitable block for size {size}")

    def free(self, offset: int) -> None:
        idx = self._find_block_index(offset)
        if idx is None:
            raise AllocationError(
                f"Invalid free: offset {offset} not allocated"
            )
        self._blocks[idx].is_free = True
        self._coalesce()

    def stats(self) -> AllocatorStats:
        allocated = sum(b.size for b in self._blocks if not b.is_free)
        free = sum(b.size for b in self._blocks if b.is_free)
        num_allocs = sum(1 for b in self._blocks if not b.is_free)
        free_blocks = [b for b in self._blocks if b.is_free]
        num_free = len(free_blocks)

        if free == 0:
            frag = 0.0
        else:
            largest_free = max((b.size for b in free_blocks), default=0)
            frag = 1.0 - (largest_free / free) if free > 0 else 0.0

        return AllocatorStats(
            total_size=self.pool_size,
            allocated=allocated,
            free=free,
            num_allocations=num_allocs,
            num_free_blocks=num_free,
            fragmentation=frag,
        )

    def dump(self) -> str:
        lines: list[str] = [f"FirstFitAllocator(pool={self.pool_size})"]
        for b in self._blocks:
            state = "FREE" if b.is_free else "USED"
            lines.append(
                f"  [{b.offset}:{b.offset + b.size}] {state} (size={b.size})"
            )
        return "\n".join(lines)
