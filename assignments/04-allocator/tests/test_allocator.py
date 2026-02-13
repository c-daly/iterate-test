import math
import random

import pytest

from allocator import Allocator, AllocationError, AllocatorStats
from buddy import BuddyAllocator


class TestBuddyAllocator:
    def test_buddy_single_alloc(self):
        """alloc(100) from 1024-byte pool returns offset 0."""
        ba = BuddyAllocator(1024)
        offset = ba.alloc(100)
        assert offset == 0

    def test_buddy_power_of_2_rounding(self):
        """alloc(50) rounds up to 64 bytes."""
        ba = BuddyAllocator(1024)
        ba.alloc(50)
        s = ba.stats()
        assert s.allocated == 64

    def test_buddy_multiple_allocs(self):
        """Three allocations return different offsets."""
        ba = BuddyAllocator(1024)
        offsets = {ba.alloc(32) for _ in range(3)}
        assert len(offsets) == 3

    def test_buddy_free_and_reuse(self):
        """Alloc, free, alloc same size reuses offset."""
        ba = BuddyAllocator(1024)
        off1 = ba.alloc(32)
        ba.free(off1)
        off2 = ba.alloc(32)
        assert off1 == off2

    def test_buddy_coalesce(self):
        """Alloc two buddies, free both, verify merge into parent block."""
        ba = BuddyAllocator(64)
        a = ba.alloc(16)
        b = ba.alloc(16)
        # a and b should be buddies (differ by 16)
        assert abs(a - b) == 16
        ba.free(a)
        ba.free(b)
        s = ba.stats()
        # After coalescing, should have fewer free blocks
        # With a 64-byte pool and min_block=16, freeing both 16-byte buddies
        # should coalesce up. Let's check we end up with 1 free block (full pool).
        assert s.num_free_blocks == 1
        assert s.free == 64

    def test_buddy_recursive_coalesce(self):
        """4 blocks from 64-byte pool, free all, full merge to single block."""
        ba = BuddyAllocator(64)
        # min_block=16, so 64/16 = 4 blocks max
        offsets = [ba.alloc(16) for _ in range(4)]
        assert len(set(offsets)) == 4
        for off in offsets:
            ba.free(off)
        s = ba.stats()
        assert s.num_free_blocks == 1
        assert s.free == 64
        assert s.allocated == 0

    def test_buddy_alloc_failure(self):
        """Filling the pool then allocating again raises AllocationError."""
        ba = BuddyAllocator(64)
        # Fill up: 64 / 16 = 4 blocks
        for _ in range(4):
            ba.alloc(16)
        with pytest.raises(AllocationError):
            ba.alloc(16)

    def test_buddy_double_free(self):
        """Freeing the same offset twice raises AllocationError."""
        ba = BuddyAllocator(1024)
        off = ba.alloc(32)
        ba.free(off)
        with pytest.raises(AllocationError):
            ba.free(off)

    def test_buddy_invalid_free(self):
        """Freeing an offset that was never allocated raises AllocationError."""
        ba = BuddyAllocator(1024)
        with pytest.raises(AllocationError):
            ba.free(999)

    def test_buddy_stats(self):
        """Check all stats fields after allocations."""
        ba = BuddyAllocator(1024)
        ba.alloc(100)  # rounds to 128
        ba.alloc(32)
        s = ba.stats()
        assert s.total_size == 1024
        assert s.allocated == 128 + 32
        assert s.free == 1024 - 128 - 32
        assert s.num_allocations == 2
        assert s.num_free_blocks > 0
        assert 0.0 <= s.fragmentation <= 1.0

    def test_buddy_fragmentation(self):
        """Alternating alloc/free creates fragmentation > 0."""
        ba = BuddyAllocator(256)
        # Allocate 4 blocks of minimum size
        offsets = [ba.alloc(16) for _ in range(4)]
        # Free alternating blocks to create fragmentation
        ba.free(offsets[0])
        ba.free(offsets[2])
        s = ba.stats()
        # Two non-adjacent free blocks of size 16, total free = 32
        # largest free block = 16, fragmentation = 1 - 16/32 = 0.5
        assert s.fragmentation > 0

    def test_buddy_stress(self):
        """200 random alloc/free operations on 4096-byte pool."""
        ba = BuddyAllocator(4096)
        allocated = []
        rng = random.Random(42)

        for _ in range(200):
            if allocated and rng.random() < 0.4:
                # Free a random allocation
                idx = rng.randint(0, len(allocated) - 1)
                ba.free(allocated.pop(idx))
            else:
                # Try to allocate
                size = rng.choice([16, 32, 64, 128, 256])
                try:
                    off = ba.alloc(size)
                    allocated.append(off)
                except AllocationError:
                    pass

        # Free everything remaining
        for off in allocated:
            ba.free(off)

        s = ba.stats()
        assert s.allocated == 0
        assert s.free == 4096
        assert s.num_free_blocks == 1  # fully coalesced


class TestAllocatorBase:
    def test_pool_size_not_power_of_2(self):
        """Non-power-of-2 pool_size raises ValueError."""
        with pytest.raises(ValueError):
            BuddyAllocator(100)

    def test_alloc_zero_raises(self):
        """alloc(0) raises AllocationError."""
        ba = BuddyAllocator(1024)
        with pytest.raises(AllocationError):
            ba.alloc(0)

    def test_alloc_negative_raises(self):
        """alloc(-1) raises AllocationError."""
        ba = BuddyAllocator(1024)
        with pytest.raises(AllocationError):
            ba.alloc(-1)

    def test_dump_returns_string(self):
        """dump() returns a string representation."""
        ba = BuddyAllocator(1024)
        result = ba.dump()
        assert isinstance(result, str)

    def test_is_abstract(self):
        """Allocator is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Allocator(1024)
