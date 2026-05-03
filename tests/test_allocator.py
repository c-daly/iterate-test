"""Tests for buddy and first-fit allocators."""
import random

import pytest

from src.allocator import AllocationError, Allocator, AllocatorStats
from src.buddy import BuddyAllocator
from src.firstfit import FirstFitAllocator
from src.compare import compare, run_workload


# ---------- Base class ----------

def test_base_class_is_abstract():
    with pytest.raises(TypeError):
        Allocator(64)  # type: ignore[abstract]


def test_stats_dataclass_fields():
    s = AllocatorStats(
        total_size=64,
        allocated=16,
        free=48,
        num_allocations=1,
        num_free_blocks=1,
        fragmentation=0.0,
    )
    assert s.total_size == 64
    assert s.allocated == 16
    assert s.free == 48
    assert s.num_allocations == 1
    assert s.num_free_blocks == 1
    assert s.fragmentation == 0.0


# ---------- Buddy allocator ----------

def test_buddy_requires_power_of_two_pool():
    with pytest.raises(ValueError):
        BuddyAllocator(100)
    BuddyAllocator(64)  # ok


def test_buddy_single_alloc_free():
    a = BuddyAllocator(64)
    off = a.alloc(8)
    assert off == 0
    s = a.stats()
    assert s.allocated >= 8
    assert s.num_allocations == 1
    a.free(off)
    s2 = a.stats()
    assert s2.allocated == 0
    assert s2.num_allocations == 0
    assert s2.free == 64


def test_buddy_rounds_up_to_power_of_two():
    a = BuddyAllocator(64)
    a.alloc(5)  # rounds up to 8
    s = a.stats()
    assert s.allocated == 8


def test_buddy_split_blocks():
    a = BuddyAllocator(64)
    o1 = a.alloc(8)
    o2 = a.alloc(8)
    assert o1 != o2
    # buddy of offset 0 with size 8 is 8
    assert {o1, o2} == {0, 8}


def test_buddy_recursive_coalesce():
    a = BuddyAllocator(64)
    offsets = [a.alloc(8) for _ in range(8)]
    # Pool fully allocated as 8 blocks of size 8
    assert sum(1 for _ in offsets) == 8
    for o in offsets:
        a.free(o)
    s = a.stats()
    assert s.allocated == 0
    assert s.free == 64
    # All buddies coalesced back into a single 64-byte free block
    assert s.num_free_blocks == 1


def test_buddy_alloc_failure_raises():
    a = BuddyAllocator(16)
    a.alloc(16)
    with pytest.raises(AllocationError):
        a.alloc(1)


def test_buddy_alloc_too_large_raises():
    a = BuddyAllocator(16)
    with pytest.raises(AllocationError):
        a.alloc(32)


def test_buddy_double_free_raises():
    a = BuddyAllocator(64)
    o = a.alloc(8)
    a.free(o)
    with pytest.raises(AllocationError):
        a.free(o)


def test_buddy_invalid_free_raises():
    a = BuddyAllocator(64)
    with pytest.raises(AllocationError):
        a.free(0)
    with pytest.raises(AllocationError):
        a.free(123456)


def test_buddy_dump_returns_string():
    a = BuddyAllocator(64)
    a.alloc(8)
    out = a.dump()
    assert isinstance(out, str)
    assert len(out) > 0


# ---------- First-fit allocator ----------

def test_firstfit_single_alloc_free():
    a = FirstFitAllocator(128)
    off = a.alloc(20)
    assert off == 0
    s = a.stats()
    assert s.allocated == 20
    assert s.num_allocations == 1
    a.free(off)
    s2 = a.stats()
    assert s2.allocated == 0
    assert s2.num_allocations == 0
    assert s2.free == 128


def test_firstfit_split():
    a = FirstFitAllocator(128)
    o1 = a.alloc(32)
    o2 = a.alloc(32)
    assert o1 == 0
    assert o2 == 32
    s = a.stats()
    assert s.allocated == 64
    assert s.free == 64


def test_firstfit_no_split_when_remainder_too_small():
    # Request leaves <16 byte remainder => should consume whole block
    a = FirstFitAllocator(64)
    o = a.alloc(50)  # remainder = 14, < 16, so whole 64 used
    assert o == 0
    s = a.stats()
    assert s.allocated == 64
    assert s.free == 0
    assert s.num_free_blocks == 0


def test_firstfit_split_when_remainder_exact_threshold():
    a = FirstFitAllocator(64)
    o = a.alloc(48)  # remainder = 16, >= 16, splits
    assert o == 0
    s = a.stats()
    assert s.allocated == 48
    assert s.free == 16


def test_firstfit_coalesce_adjacent():
    a = FirstFitAllocator(128)
    o1 = a.alloc(32)
    o2 = a.alloc(32)
    o3 = a.alloc(32)
    a.free(o1)
    a.free(o3)
    # Two non-adjacent free blocks: [0..32) and [64..96), and [96..128)
    s_mid = a.stats()
    # free block count: [0..32) free, [64..96) free, [96..128) free => but [64..96) and [96..128) are adjacent and should coalesce
    assert s_mid.num_free_blocks == 2
    a.free(o2)
    s = a.stats()
    assert s.num_free_blocks == 1
    assert s.free == 128
    assert s.allocated == 0


def test_firstfit_alloc_failure_raises():
    a = FirstFitAllocator(64)
    a.alloc(64)
    with pytest.raises(AllocationError):
        a.alloc(1)


def test_firstfit_alloc_too_large_raises():
    a = FirstFitAllocator(64)
    with pytest.raises(AllocationError):
        a.alloc(128)


def test_firstfit_double_free_raises():
    a = FirstFitAllocator(64)
    o = a.alloc(20)
    a.free(o)
    with pytest.raises(AllocationError):
        a.free(o)


def test_firstfit_invalid_free_raises():
    a = FirstFitAllocator(64)
    with pytest.raises(AllocationError):
        a.free(0)
    with pytest.raises(AllocationError):
        a.free(99999)


def test_firstfit_dump_returns_string():
    a = FirstFitAllocator(128)
    a.alloc(32)
    out = a.dump()
    assert isinstance(out, str)
    assert len(out) > 0


# ---------- Fragmentation ----------

def test_fragmentation_zero_when_single_free_block():
    a = FirstFitAllocator(128)
    s = a.stats()
    assert s.fragmentation == 0.0


def test_fragmentation_positive_when_split():
    a = FirstFitAllocator(128)
    o1 = a.alloc(32)
    o2 = a.alloc(32)
    o3 = a.alloc(32)
    a.free(o2)  # creates a hole
    # remaining free: [64..96) of 32 bytes; allocated rest. Wait o3 still allocated.
    # Actually: free regions = middle 32 + tail 32. Two 32-byte blocks.
    s = a.stats()
    assert s.fragmentation > 0
    # cleanup-ish use of o1, o3 to silence linters
    assert o1 == 0
    assert o3 == 64


def test_fragmentation_zero_when_all_free():
    a = FirstFitAllocator(128)
    o = a.alloc(32)
    a.free(o)
    s = a.stats()
    assert s.fragmentation == 0.0


# ---------- Stress test ----------

@pytest.mark.parametrize("alloc_cls", [BuddyAllocator, FirstFitAllocator])
def test_random_stress(alloc_cls):
    rng = random.Random(1234)
    a = alloc_cls(1024)
    live: list[tuple[int, int]] = []  # (offset, size)
    for _ in range(500):
        if live and rng.random() < 0.5:
            i = rng.randrange(len(live))
            off, _ = live.pop(i)
            a.free(off)
        else:
            size = rng.randint(1, 64)
            try:
                off = a.alloc(size)
                live.append((off, size))
            except AllocationError:
                pass
    # free remaining
    for off, _ in live:
        a.free(off)
    s = a.stats()
    assert s.allocated == 0
    assert s.num_allocations == 0
    assert s.free == 1024


# ---------- Compare ----------

def test_run_workload_returns_stats():
    a = FirstFitAllocator(128)
    ops: list[tuple] = [("alloc", 16), ("alloc", 16), ("free", 0)]
    stats = run_workload(a, ops)
    assert isinstance(stats, AllocatorStats)
    assert stats.allocated == 16
    assert stats.num_allocations == 1


def test_compare_mixed_workload():
    ops: list[tuple] = [
        ("alloc", 16),
        ("alloc", 32),
        ("alloc", 8),
        ("free", 0),
        ("alloc", 16),
    ]
    result = compare(128, ops)
    assert "buddy" in result
    assert "firstfit" in result
    assert isinstance(result["buddy"], AllocatorStats)
    assert isinstance(result["firstfit"], AllocatorStats)
