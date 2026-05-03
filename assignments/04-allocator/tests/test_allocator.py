"""Tests for the buddy and first-fit allocators and the comparison harness.

Covers single alloc/free, power-of-2 rounding, splitting, coalescing
(including recursive buddy coalescing), allocation failure, fragmentation,
double/invalid free, randomized stress, and the comparison helper.
"""
from __future__ import annotations

import random

import pytest

from src.allocator import AllocationError, Allocator, AllocatorStats
from src.buddy import BuddyAllocator
from src.compare import compare, run_workload
from src.firstfit import FirstFitAllocator


# ---------------------------------------------------------------------------
# Base class / dataclass
# ---------------------------------------------------------------------------


def test_stats_dataclass_fields() -> None:
    s = AllocatorStats(
        total_size=1024,
        allocated=128,
        free=896,
        num_allocations=2,
        num_free_blocks=3,
        fragmentation=0.25,
    )
    assert s.total_size == 1024
    assert s.allocated == 128
    assert s.free == 896
    assert s.num_allocations == 2
    assert s.num_free_blocks == 3
    assert s.fragmentation == pytest.approx(0.25)


def test_allocator_is_abstract() -> None:
    with pytest.raises(TypeError):
        Allocator(1024)  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Buddy allocator
# ---------------------------------------------------------------------------


def test_buddy_requires_power_of_two_pool() -> None:
    with pytest.raises(ValueError):
        BuddyAllocator(1000)


def test_buddy_single_alloc_free() -> None:
    a = BuddyAllocator(1024)
    off = a.alloc(64)
    assert isinstance(off, int)
    s = a.stats()
    assert s.total_size == 1024
    assert s.allocated == 64
    assert s.num_allocations == 1
    a.free(off)
    s2 = a.stats()
    assert s2.allocated == 0
    assert s2.num_allocations == 0
    assert s2.free == 1024


def test_buddy_rounds_up_to_power_of_two() -> None:
    a = BuddyAllocator(1024)
    a.alloc(33)  # rounds up to 64
    s = a.stats()
    assert s.allocated == 64


def test_buddy_block_split_and_coalesce() -> None:
    a = BuddyAllocator(256)
    o1 = a.alloc(64)
    o2 = a.alloc(64)
    # buddies of each other if both came from same parent split
    assert (o1 ^ 64) == o2 or (o2 ^ 64) == o1
    a.free(o1)
    a.free(o2)
    s = a.stats()
    # full coalesce back to single 256 free block
    assert s.allocated == 0
    assert s.num_free_blocks == 1


def test_buddy_recursive_coalescing() -> None:
    """Free 4 adjacent 16-byte buddies and verify coalesce climbs all the way up."""
    a = BuddyAllocator(64)
    offs = [a.alloc(16) for _ in range(4)]
    for o in offs:
        a.free(o)
    s = a.stats()
    assert s.allocated == 0
    # All four 16s should have merged into a single 64-byte free block.
    assert s.num_free_blocks == 1
    assert s.fragmentation == pytest.approx(0.0)


def test_buddy_allocation_failure_raises() -> None:
    a = BuddyAllocator(64)
    a.alloc(64)
    with pytest.raises(AllocationError):
        a.alloc(16)


def test_buddy_alloc_too_large_raises() -> None:
    a = BuddyAllocator(64)
    with pytest.raises(AllocationError):
        a.alloc(128)


def test_buddy_alloc_zero_or_negative_raises() -> None:
    a = BuddyAllocator(64)
    with pytest.raises(AllocationError):
        a.alloc(0)
    with pytest.raises(AllocationError):
        a.alloc(-8)


def test_buddy_double_free_raises() -> None:
    a = BuddyAllocator(64)
    o = a.alloc(16)
    a.free(o)
    with pytest.raises(AllocationError):
        a.free(o)


def test_buddy_invalid_free_raises() -> None:
    a = BuddyAllocator(64)
    with pytest.raises(AllocationError):
        a.free(7)  # never allocated, not aligned
    with pytest.raises(AllocationError):
        a.free(9999)  # out of range


def test_buddy_fragmentation_zero_when_full() -> None:
    a = BuddyAllocator(64)
    a.alloc(64)
    s = a.stats()
    # No free blocks at all: fragmentation defined as 0.0.
    assert s.fragmentation == pytest.approx(0.0)


def test_buddy_dump_is_string() -> None:
    a = BuddyAllocator(64)
    a.alloc(16)
    out = a.dump()
    assert isinstance(out, str)
    assert "\n" in out or len(out) > 0


# ---------------------------------------------------------------------------
# First-fit allocator
# ---------------------------------------------------------------------------


def test_firstfit_single_alloc_free() -> None:
    a = FirstFitAllocator(1024)
    off = a.alloc(100)
    assert isinstance(off, int)
    s = a.stats()
    assert s.allocated == 100
    assert s.num_allocations == 1
    a.free(off)
    s2 = a.stats()
    assert s2.allocated == 0
    assert s2.num_allocations == 0
    assert s2.free == 1024
    assert s2.num_free_blocks == 1


def test_firstfit_split_when_remainder_large() -> None:
    a = FirstFitAllocator(1024)
    off = a.alloc(100)
    assert off == 0
    s = a.stats()
    # remainder 924 >= 16, so we get exactly one free block of 924.
    assert s.num_free_blocks == 1
    assert s.free == 924


def test_firstfit_no_split_when_remainder_small() -> None:
    a = FirstFitAllocator(64)
    a.alloc(50)  # remainder 14 < 16, take whole 64.
    s = a.stats()
    # Even though the whole 64-byte block is consumed, the caller asked for 50.
    # ``allocated`` is requested-size accounting; the absorbed 14 bytes show up
    # as missing free space (free == 0, no split block) but never inflate
    # ``allocated`` past what callers requested.
    assert s.allocated == 50
    assert s.num_free_blocks == 0
    assert s.free == 0


def test_firstfit_allocated_reflects_requested_not_padded() -> None:
    """alloc(120) from a 128 pool absorbs the 8-byte tail (< split threshold 16);
    stats.allocated must report the requested 120, not the padded 128."""
    a = FirstFitAllocator(128)
    off = a.alloc(120)
    s = a.stats()
    assert s.allocated == 120
    assert s.num_allocations == 1
    # The 8-byte tail is absorbed (no split): no free block survives.
    assert s.num_free_blocks == 0
    assert s.free == 0
    # On free, the allocator must recover the true 128-byte block size from
    # the neighbours and return all of it to the pool.
    a.free(off)
    s2 = a.stats()
    assert s2.allocated == 0
    assert s2.free == 128
    assert s2.num_free_blocks == 1


def test_firstfit_coalesce_left_right_neighbors() -> None:
    a = FirstFitAllocator(1024)
    o1 = a.alloc(100)
    o2 = a.alloc(100)
    o3 = a.alloc(100)
    a.free(o1)
    a.free(o3)
    s_mid = a.stats()
    # three free blocks: [0:100], gap, [200:300] no -- check a more general invariant
    assert s_mid.num_free_blocks >= 2
    a.free(o2)
    s = a.stats()
    # all three coalesce into a single free block of 1024.
    assert s.allocated == 0
    assert s.num_free_blocks == 1
    assert s.free == 1024


def test_firstfit_allocation_failure_raises() -> None:
    a = FirstFitAllocator(128)
    a.alloc(128)
    with pytest.raises(AllocationError):
        a.alloc(1)


def test_firstfit_alloc_too_large_raises() -> None:
    a = FirstFitAllocator(64)
    with pytest.raises(AllocationError):
        a.alloc(128)


def test_firstfit_alloc_zero_or_negative_raises() -> None:
    a = FirstFitAllocator(64)
    with pytest.raises(AllocationError):
        a.alloc(0)
    with pytest.raises(AllocationError):
        a.alloc(-1)


def test_firstfit_double_free_raises() -> None:
    a = FirstFitAllocator(128)
    o = a.alloc(32)
    a.free(o)
    with pytest.raises(AllocationError):
        a.free(o)


def test_firstfit_invalid_free_raises() -> None:
    a = FirstFitAllocator(128)
    with pytest.raises(AllocationError):
        a.free(9999)
    with pytest.raises(AllocationError):
        a.free(50)  # never allocated


def test_firstfit_fragmentation_metric() -> None:
    """With a single free block, fragmentation == 0; with multiple, > 0."""
    a = FirstFitAllocator(1024)
    o1 = a.alloc(100)
    o2 = a.alloc(100)
    o3 = a.alloc(100)
    a.free(o1)
    a.free(o3)  # leaves two non-adjacent free blocks plus the trailing tail
    s = a.stats()
    assert s.num_free_blocks >= 2
    assert s.fragmentation > 0.0
    assert 0.0 <= s.fragmentation <= 1.0
    # use o2 to silence unused-var if any linter complains
    assert isinstance(o2, int)


def test_firstfit_dump_is_string() -> None:
    a = FirstFitAllocator(64)
    a.alloc(16)
    out = a.dump()
    assert isinstance(out, str)


# ---------------------------------------------------------------------------
# Stress tests (random alloc/free)
# ---------------------------------------------------------------------------


def _stress(allocator: Allocator, *, iterations: int, max_size: int, seed: int) -> None:
    rng = random.Random(seed)
    live: list[tuple[int, int]] = []  # (offset, size) for invariant checks
    for _ in range(iterations):
        if live and rng.random() < 0.5:
            idx = rng.randrange(len(live))
            off, _sz = live.pop(idx)
            allocator.free(off)
        else:
            sz = rng.randint(1, max_size)
            try:
                off = allocator.alloc(sz)
            except AllocationError:
                continue
            live.append((off, sz))
    # free everything still live; allocator should remain consistent.
    for off, _sz in live:
        allocator.free(off)
    s = allocator.stats()
    assert s.allocated == 0
    assert s.num_allocations == 0
    assert s.free == s.total_size


def test_buddy_stress_random_alloc_free() -> None:
    a = BuddyAllocator(1024)
    _stress(a, iterations=500, max_size=128, seed=42)
    # After freeing everything, buddy should fully coalesce.
    assert a.stats().num_free_blocks == 1


def test_firstfit_stress_random_alloc_free() -> None:
    a = FirstFitAllocator(2048)
    _stress(a, iterations=500, max_size=256, seed=7)
    assert a.stats().num_free_blocks == 1


# ---------------------------------------------------------------------------
# Compare harness
# ---------------------------------------------------------------------------


def test_run_workload_returns_stats() -> None:
    a = BuddyAllocator(1024)
    ops: list[tuple] = [("alloc", 64), ("alloc", 32), ("free", 0)]
    out = run_workload(a, ops)
    assert isinstance(out, AllocatorStats)
    assert out.total_size == 1024


def test_compare_returns_dict_with_both_allocators() -> None:
    ops: list[tuple] = [
        ("alloc", 64),
        ("alloc", 128),
        ("alloc", 32),
        ("free", 1),
        ("alloc", 16),
        ("free", 0),
    ]
    result = compare(1024, ops)
    assert isinstance(result, dict)
    assert "buddy" in result
    assert "firstfit" in result
    assert isinstance(result["buddy"], AllocatorStats)
    assert isinstance(result["firstfit"], AllocatorStats)
    assert result["buddy"].total_size == 1024
    assert result["firstfit"].total_size == 1024


def test_run_workload_free_by_op_index() -> None:
    """``free`` ops reference the *index of the alloc op* whose offset to release."""
    a = FirstFitAllocator(1024)
    ops: list[tuple] = [
        ("alloc", 100),  # index 0
        ("alloc", 100),  # index 1
        ("free", 0),
        ("free", 1),
    ]
    out = run_workload(a, ops)
    assert out.allocated == 0
    assert out.num_allocations == 0
