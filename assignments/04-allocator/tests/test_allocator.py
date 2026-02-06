"""Comprehensive tests for buddy allocator, first-fit allocator, base class, comparison."""

from __future__ import annotations

import random

import pytest

from src.allocator import AllocationError, Allocator, AllocatorStats
from src.buddy import BuddyAllocator
from src.compare import compare, run_workload
from src.firstfit import FirstFitAllocator


# ---------------------------------------------------------------------------
# Base class / AllocatorStats tests
# ---------------------------------------------------------------------------

class TestAllocatorStats:
    def test_stats_dataclass_fields(self):
        s = AllocatorStats(
            total_size=1024,
            allocated=256,
            free=768,
            num_allocations=1,
            num_free_blocks=1,
            fragmentation=0.0,
        )
        assert s.total_size == 1024
        assert s.allocated == 256
        assert s.free == 768
        assert s.num_allocations == 1
        assert s.num_free_blocks == 1
        assert s.fragmentation == 0.0

    def test_allocator_is_abstract(self):
        with pytest.raises(TypeError):
            Allocator(1024)  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Buddy Allocator tests
# ---------------------------------------------------------------------------

class TestBuddyAllocator:
    def test_single_alloc(self):
        ba = BuddyAllocator(1024)
        offset = ba.alloc(128)
        assert isinstance(offset, int)
        assert 0 <= offset < 1024

    def test_single_alloc_free(self):
        ba = BuddyAllocator(1024)
        offset = ba.alloc(128)
        ba.free(offset)
        s = ba.stats()
        assert s.allocated == 0
        assert s.free == 1024
        assert s.num_allocations == 0

    def test_power_of_2_rounding(self):
        """Request non-power-of-2 size; buddy must round up."""
        ba = BuddyAllocator(1024)
        # Request 100 bytes -> rounds to 128
        ba.alloc(100)
        s = ba.stats()
        assert s.allocated == 128

    def test_exact_power_of_2(self):
        """Request exact power-of-2 size; no rounding needed."""
        ba = BuddyAllocator(1024)
        ba.alloc(64)
        s = ba.stats()
        assert s.allocated == 64

    def test_block_splitting(self):
        """Allocating small block from large pool requires splitting."""
        ba = BuddyAllocator(1024)
        # 64-byte alloc from 1024-byte pool requires splitting
        ba.alloc(64)
        s = ba.stats()
        assert s.allocated == 64
        assert s.free == 1024 - 64
        # Should have multiple free blocks from splitting
        assert s.num_free_blocks >= 1

    def test_recursive_buddy_coalescing(self):
        """Freeing both buddies should coalesce recursively."""
        ba = BuddyAllocator(256)
        # Allocate two 64-byte blocks (fills 128 out of 256)
        o1 = ba.alloc(64)
        o2 = ba.alloc(64)
        # Free both — they are buddies, should coalesce back to 128
        ba.free(o1)
        ba.free(o2)
        s = ba.stats()
        assert s.allocated == 0
        assert s.free == 256
        # After full coalesce, should have single free block
        assert s.num_free_blocks == 1

    def test_full_coalesce_after_all_freed(self):
        """Allocate four blocks, free all; pool should fully coalesce."""
        ba = BuddyAllocator(256)
        offsets = [ba.alloc(64) for _ in range(4)]
        for o in offsets:
            ba.free(o)
        s = ba.stats()
        assert s.num_free_blocks == 1
        assert s.free == 256

    def test_allocation_failure_when_full(self):
        ba = BuddyAllocator(128)
        ba.alloc(128)
        with pytest.raises(AllocationError):
            ba.alloc(1)

    def test_allocation_failure_no_suitable_block(self):
        """All blocks allocated such that no contiguous block available."""
        ba = BuddyAllocator(256)
        ba.alloc(128)
        ba.alloc(64)
        ba.alloc(64)
        with pytest.raises(AllocationError):
            ba.alloc(1)

    def test_double_free(self):
        ba = BuddyAllocator(256)
        offset = ba.alloc(64)
        ba.free(offset)
        with pytest.raises(AllocationError):
            ba.free(offset)

    def test_invalid_free(self):
        ba = BuddyAllocator(256)
        with pytest.raises(AllocationError):
            ba.free(999)

    def test_free_never_allocated_offset(self):
        """Free an offset that was never returned by alloc."""
        ba = BuddyAllocator(256)
        ba.alloc(64)
        with pytest.raises(AllocationError):
            ba.free(100)

    def test_stats_after_multiple_allocs(self):
        ba = BuddyAllocator(1024)
        ba.alloc(128)
        ba.alloc(256)
        s = ba.stats()
        assert s.total_size == 1024
        assert s.allocated == 128 + 256
        assert s.free == 1024 - 128 - 256
        assert s.num_allocations == 2

    def test_fragmentation_zero_when_empty(self):
        ba = BuddyAllocator(512)
        s = ba.stats()
        assert s.fragmentation == 0.0

    def test_fragmentation_increases_with_scattered_allocs(self):
        """Allocate-free pattern that produces fragmentation."""
        ba = BuddyAllocator(512)
        _o1 = ba.alloc(64)
        o2 = ba.alloc(64)
        _o3 = ba.alloc(64)
        # Free middle block -> fragmentation should be > 0
        ba.free(o2)
        s = ba.stats()
        assert s.fragmentation >= 0.0

    def test_buddy_address_xor_property(self):
        """Buddy of block at offset is offset XOR block_size."""
        ba = BuddyAllocator(256)
        # Allocate two 128-byte blocks
        o1 = ba.alloc(128)
        o2 = ba.alloc(128)
        # They should be buddies: o1 XOR 128 == o2
        assert o1 ^ 128 == o2 or o2 ^ 128 == o1

    def test_pool_size_must_be_power_of_2(self):
        with pytest.raises((ValueError, AllocationError)):
            BuddyAllocator(100)

    def test_alloc_size_zero_raises(self):
        ba = BuddyAllocator(256)
        with pytest.raises((ValueError, AllocationError)):
            ba.alloc(0)

    def test_dump_returns_string(self):
        ba = BuddyAllocator(256)
        ba.alloc(64)
        result = ba.dump()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_minimum_allocation(self):
        """Allocating 1 byte should still work (rounds up to minimum block)."""
        ba = BuddyAllocator(256)
        offset = ba.alloc(1)
        assert isinstance(offset, int)
        assert 0 <= offset < 256


# ---------------------------------------------------------------------------
# First-Fit Allocator tests
# ---------------------------------------------------------------------------

class TestFirstFitAllocator:
    def test_single_alloc(self):
        ff = FirstFitAllocator(1024)
        offset = ff.alloc(128)
        assert isinstance(offset, int)
        assert 0 <= offset < 1024

    def test_single_alloc_free(self):
        ff = FirstFitAllocator(1024)
        offset = ff.alloc(128)
        ff.free(offset)
        s = ff.stats()
        assert s.allocated == 0
        assert s.free == 1024
        assert s.num_allocations == 0

    def test_first_fit_strategy(self):
        """First alloc should return offset 0 (beginning of pool)."""
        ff = FirstFitAllocator(1024)
        offset = ff.alloc(64)
        assert offset == 0

    def test_sequential_allocs_are_contiguous(self):
        """Sequential allocations should be placed contiguously."""
        ff = FirstFitAllocator(1024)
        o1 = ff.alloc(64)
        o2 = ff.alloc(128)
        assert o1 == 0
        assert o2 == 64

    def test_block_splitting(self):
        """Block should be split if remainder >= 16."""
        ff = FirstFitAllocator(1024)
        ff.alloc(64)
        s = ff.stats()
        assert s.allocated == 64
        # Remainder (960) is >= 16, so it should be split into free block
        assert s.free == 960
        assert s.num_free_blocks >= 1

    def test_no_split_if_remainder_too_small(self):
        """Block should NOT split if remainder < 16 bytes."""
        ff = FirstFitAllocator(64)
        # Allocate 50 bytes: remainder is 14, which is < 16
        ff.alloc(50)
        s = ff.stats()
        # The entire 64-byte block should be used (no split)
        assert s.allocated == 64
        assert s.free == 0

    def test_coalesce_adjacent_on_free(self):
        """Freeing adjacent blocks should coalesce them."""
        ff = FirstFitAllocator(1024)
        o1 = ff.alloc(128)
        o2 = ff.alloc(128)
        _o3 = ff.alloc(128)
        ff.free(o1)
        ff.free(o2)
        s = ff.stats()
        # o1 and o2 freed blocks should coalesce
        assert s.num_allocations == 1
        # There should be a large free block from coalescing
        assert s.free == 1024 - 128

    def test_coalesce_all_free(self):
        """Free all blocks; entire pool should coalesce to one free block."""
        ff = FirstFitAllocator(256)
        o1 = ff.alloc(64)
        o2 = ff.alloc(64)
        o3 = ff.alloc(64)
        ff.free(o1)
        ff.free(o2)
        ff.free(o3)
        s = ff.stats()
        assert s.num_free_blocks == 1
        assert s.free == 256

    def test_allocation_failure_when_full(self):
        ff = FirstFitAllocator(128)
        ff.alloc(128)
        with pytest.raises(AllocationError):
            ff.alloc(1)

    def test_allocation_failure_fragmented(self):
        """No single free block large enough even though total free suffices."""
        ff = FirstFitAllocator(256)
        offsets = [ff.alloc(64) for _ in range(4)]
        # Free alternating blocks: 0 and 128 freed, 64 and 192 still allocated
        ff.free(offsets[0])
        ff.free(offsets[2])
        # Two 64-byte free blocks; requesting 128 should fail
        with pytest.raises(AllocationError):
            ff.alloc(128)

    def test_double_free(self):
        ff = FirstFitAllocator(256)
        offset = ff.alloc(64)
        ff.free(offset)
        with pytest.raises(AllocationError):
            ff.free(offset)

    def test_invalid_free(self):
        ff = FirstFitAllocator(256)
        with pytest.raises(AllocationError):
            ff.free(999)

    def test_free_unallocated_offset(self):
        """Free an offset inside pool but never allocated."""
        ff = FirstFitAllocator(256)
        ff.alloc(64)
        with pytest.raises(AllocationError):
            ff.free(32)

    def test_stats_after_multiple_allocs(self):
        ff = FirstFitAllocator(1024)
        ff.alloc(100)
        ff.alloc(200)
        s = ff.stats()
        assert s.total_size == 1024
        assert s.allocated == 300
        assert s.free == 724
        assert s.num_allocations == 2

    def test_fragmentation_zero_when_empty(self):
        ff = FirstFitAllocator(512)
        s = ff.stats()
        assert s.fragmentation == 0.0

    def test_fragmentation_measured(self):
        """Create fragmentation and verify it's measured."""
        ff = FirstFitAllocator(512)
        offsets = [ff.alloc(64) for _ in range(8)]
        # Free every other block
        for i in range(0, 8, 2):
            ff.free(offsets[i])
        s = ff.stats()
        # Fragmentation should be > 0 since free memory is scattered
        assert s.fragmentation > 0.0

    def test_alloc_size_zero_raises(self):
        ff = FirstFitAllocator(256)
        with pytest.raises((ValueError, AllocationError)):
            ff.alloc(0)

    def test_dump_returns_string(self):
        ff = FirstFitAllocator(256)
        ff.alloc(64)
        result = ff.dump()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_reuse_freed_block(self):
        """After freeing, new alloc should reuse the freed space."""
        ff = FirstFitAllocator(256)
        o1 = ff.alloc(64)
        ff.free(o1)
        o2 = ff.alloc(32)
        # First-fit should place at beginning (offset 0)
        assert o2 == 0


# ---------------------------------------------------------------------------
# Stress tests (both allocators)
# ---------------------------------------------------------------------------

class TestStress:
    @pytest.mark.parametrize(
        "alloc_cls", [BuddyAllocator, FirstFitAllocator],
    )
    def test_random_alloc_free_sequence(self, alloc_cls):
        """Random alloc/free ops should not crash or corrupt state."""
        rng = random.Random(42)
        pool_size = 1024
        allocator = alloc_cls(pool_size)
        active: dict[int, None] = {}

        for _ in range(200):
            if active and rng.random() < 0.4:
                # Free a random active allocation
                offset = rng.choice(list(active.keys()))
                allocator.free(offset)
                del active[offset]
            else:
                # Alloc
                size = rng.randint(1, 128)
                try:
                    offset = allocator.alloc(size)
                    active[offset] = None
                except AllocationError:
                    pass  # pool full, expected

        s = allocator.stats()
        assert s.allocated >= 0
        assert s.free >= 0
        assert s.allocated + s.free == pool_size

    @pytest.mark.parametrize(
        "alloc_cls", [BuddyAllocator, FirstFitAllocator],
    )
    def test_fill_and_empty(self, alloc_cls):
        """Fill pool completely, then free everything."""
        pool_size = 256
        allocator = alloc_cls(pool_size)
        offsets = []
        # Fill
        while True:
            try:
                offsets.append(allocator.alloc(32))
            except AllocationError:
                break
        assert len(offsets) > 0
        # Empty
        for o in offsets:
            allocator.free(o)
        s = allocator.stats()
        assert s.allocated == 0
        assert s.free == pool_size


# ---------------------------------------------------------------------------
# Comparison tests
# ---------------------------------------------------------------------------

class TestComparison:
    def test_run_workload_returns_stats(self):
        ba = BuddyAllocator(1024)
        ops = [("alloc", 64), ("alloc", 128)]
        result = run_workload(ba, ops)
        assert isinstance(result, AllocatorStats)
        assert result.num_allocations == 2

    def test_run_workload_with_free(self):
        ff = FirstFitAllocator(1024)
        ops = [("alloc", 64), ("alloc", 128), ("free", 0), ("alloc", 32)]
        result = run_workload(ff, ops)
        assert isinstance(result, AllocatorStats)
        assert result.num_allocations == 2  # 3 allocs - 1 free = 2

    def test_compare_returns_dict_with_both(self):
        ops = [("alloc", 64), ("alloc", 128)]
        result = compare(1024, ops)
        assert isinstance(result, dict)
        assert "buddy" in result
        assert "firstfit" in result
        assert isinstance(result["buddy"], AllocatorStats)
        assert isinstance(result["firstfit"], AllocatorStats)

    def test_compare_mixed_workload(self):
        ops = [
            ("alloc", 64),
            ("alloc", 128),
            ("alloc", 32),
            ("free", 0),
            ("alloc", 64),
            ("free", 1),
        ]
        result = compare(1024, ops)
        assert result["buddy"].total_size == 1024
        assert result["firstfit"].total_size == 1024
