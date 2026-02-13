import math
import random

import pytest

from allocator import Allocator, AllocationError, AllocatorStats
from buddy import BuddyAllocator
from compare import run_workload, compare


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


from firstfit import FirstFitAllocator


class TestFirstFitAllocBasic:
    def test_ff_single_alloc(self):
        """alloc(100) from 1024 pool returns offset 0."""
        a = FirstFitAllocator(1024)
        assert a.alloc(100) == 0

    def test_ff_multiple_allocs(self):
        """Three allocs: 100, 200, 50 -> offsets 0, 100, 300."""
        a = FirstFitAllocator(1024)
        o1 = a.alloc(100)
        o2 = a.alloc(200)
        o3 = a.alloc(50)
        assert o1 == 0
        assert o2 == 100
        assert o3 == 300


class TestFirstFitFreeAndReuse:
    def test_ff_free_and_reuse(self):
        """alloc(100), free(0), alloc(100) -> offset 0 again."""
        a = FirstFitAllocator(1024)
        o1 = a.alloc(100)
        a.free(o1)
        o2 = a.alloc(100)
        assert o2 == 0

    def test_ff_coalesce(self):
        """alloc A(100), alloc B(100), free A, free B -> single free block."""
        a = FirstFitAllocator(1024)
        o1 = a.alloc(100)
        o2 = a.alloc(100)
        a.free(o1)
        a.free(o2)
        s = a.stats()
        assert s.num_free_blocks == 1
        assert s.free == 1024

    def test_ff_split_min_size(self):
        """Alloc leaves < 16 remainder -> no split (absorbed into alloc)."""
        a = FirstFitAllocator(1024)
        o = a.alloc(1010)
        assert o == 0
        s = a.stats()
        assert s.num_free_blocks == 0
        assert s.num_allocations == 1

    def test_ff_first_fit_behavior(self):
        """alloc A(100), alloc B(100), free A, alloc C(50) -> C gets offset 0."""
        a = FirstFitAllocator(1024)
        o_a = a.alloc(100)
        _o_b = a.alloc(100)
        a.free(o_a)
        o_c = a.alloc(50)
        assert o_c == 0


class TestFirstFitErrors:
    def test_ff_alloc_failure(self):
        """Fill pool, next alloc raises AllocationError."""
        a = FirstFitAllocator(256)
        a.alloc(256)
        with pytest.raises(AllocationError):
            a.alloc(1)

    def test_ff_double_free(self):
        """Freeing the same offset twice raises AllocationError."""
        a = FirstFitAllocator(1024)
        o = a.alloc(100)
        a.free(o)
        with pytest.raises(AllocationError):
            a.free(o)

    def test_ff_invalid_free(self):
        """Freeing an offset that was never allocated raises AllocationError."""
        a = FirstFitAllocator(1024)
        with pytest.raises(AllocationError):
            a.free(999)


class TestFirstFitStats:
    def test_ff_fragmentation(self):
        """Create gaps, verify fragmentation > 0."""
        a = FirstFitAllocator(1024)
        o_a = a.alloc(100)
        _o_b = a.alloc(100)
        _o_c = a.alloc(100)
        a.free(o_a)
        s = a.stats()
        assert s.fragmentation > 0


class TestFirstFitStress:
    def test_ff_stress(self):
        """200 random alloc/free ops, no crashes."""
        a = FirstFitAllocator(8192)
        rng = random.Random(42)
        live = []

        for _ in range(200):
            if not live or rng.random() < 0.6:
                size = rng.randint(1, 512)
                try:
                    offset = a.alloc(size)
                    live.append(offset)
                except AllocationError:
                    pass
            else:
                idx = rng.randint(0, len(live) - 1)
                offset = live.pop(idx)
                a.free(offset)

        s = a.stats()
        assert s.total_size == 8192
        assert s.num_allocations == len(live)
        assert 0.0 <= s.fragmentation <= 1.0


class TestCompare:
    def test_run_workload(self):
        """run_workload executes alloc+free sequence and returns stats."""
        ba = BuddyAllocator(1024)
        ops = [
            ("alloc", 64),
            ("alloc", 128),
            ("free", 0),  # free the first alloc (64 bytes)
        ]
        stats = run_workload(ba, ops)
        assert isinstance(stats, AllocatorStats)
        assert stats.total_size == 1024
        assert stats.num_allocations == 1  # one alloc freed, one remains
        assert stats.allocated == 128

    def test_compare_returns_both(self):
        """compare() returns dict with buddy and firstfit keys, both AllocatorStats."""
        ops = [("alloc", 64), ("alloc", 128)]
        result = compare(1024, ops)
        assert "buddy" in result
        assert "firstfit" in result
        assert isinstance(result["buddy"], AllocatorStats)
        assert isinstance(result["firstfit"], AllocatorStats)

    def test_compare_mixed_workload(self):
        """10 allocs and 5 frees produce valid stats from both allocators."""
        ops = [("alloc", 32 + i * 4) for i in range(10)]
        # free allocs at indices 1, 3, 5, 7, 9
        ops += [("free", i) for i in range(1, 10, 2)]
        result = compare(4096, ops)
        for key in ("buddy", "firstfit"):
            s = result[key]
            assert s.total_size == 4096
            assert s.num_allocations == 5  # 10 allocs - 5 frees
            assert s.allocated > 0
            assert s.free > 0
            assert 0.0 <= s.fragmentation <= 1.0

    def test_compare_fragmentation_difference(self):
        """Buddy rounds up to power-of-2; firstfit packs exactly -> different allocated."""
        # Allocate sizes that are NOT powers of 2 so buddy wastes space
        ops = [
            ("alloc", 33),   # buddy rounds to 64, firstfit uses 33
            ("alloc", 65),   # buddy rounds to 128, firstfit uses 65
            ("alloc", 100),  # buddy rounds to 128, firstfit uses 100
        ]
        result = compare(1024, ops)
        buddy_stats = result["buddy"]
        ff_stats = result["firstfit"]
        # Buddy allocates more total memory due to rounding
        assert buddy_stats.allocated > ff_stats.allocated
        # Both should have 3 active allocations
        assert buddy_stats.num_allocations == 3
        assert ff_stats.num_allocations == 3
        # Buddy: 64+128+128 = 320 allocated
        assert buddy_stats.allocated == 320
        # FirstFit: 33+65+100 = 198 allocated
        assert ff_stats.allocated == 198
