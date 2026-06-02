"""Tests for the memory allocator assignment.

Covers both the buddy allocator and the first-fit allocator plus the
comparison tooling. The allocators simulate a memory pool via offset
tracking -- no real memory is touched.
"""

import random

import pytest

from allocator import AllocationError, Allocator, AllocatorStats
from buddy import BuddyAllocator
from firstfit import FirstFitAllocator
from compare import compare, run_workload


ALLOCATOR_CLASSES = [BuddyAllocator, FirstFitAllocator]


# --------------------------------------------------------------------------
# Base class / stats contract
# --------------------------------------------------------------------------

def test_allocator_is_abstract():
    with pytest.raises(TypeError):
        Allocator(1024)  # type: ignore[abstract]


def test_allocator_stats_fields():
    stats = AllocatorStats(
        total_size=1024,
        allocated=0,
        free=1024,
        num_allocations=0,
        num_free_blocks=1,
        fragmentation=0.0,
    )
    assert stats.total_size == 1024
    assert stats.allocated == 0
    assert stats.free == 1024
    assert stats.num_allocations == 0
    assert stats.num_free_blocks == 1
    assert stats.fragmentation == 0.0


@pytest.mark.parametrize("cls", ALLOCATOR_CLASSES)
def test_initial_stats(cls):
    a = cls(1024)
    stats = a.stats()
    assert isinstance(stats, AllocatorStats)
    assert stats.total_size == 1024
    assert stats.allocated == 0
    assert stats.free == 1024
    assert stats.num_allocations == 0


@pytest.mark.parametrize("cls", ALLOCATOR_CLASSES)
def test_dump_returns_string(cls):
    a = cls(1024)
    a.alloc(16)
    out = a.dump()
    assert isinstance(out, str)
    assert len(out) > 0


# --------------------------------------------------------------------------
# Single alloc / free for both allocators
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALLOCATOR_CLASSES)
def test_single_alloc_returns_offset(cls):
    a = cls(1024)
    off = a.alloc(32)
    assert isinstance(off, int)
    assert 0 <= off < 1024
    stats = a.stats()
    assert stats.allocated >= 32
    assert stats.num_allocations == 1


@pytest.mark.parametrize("cls", ALLOCATOR_CLASSES)
def test_single_alloc_then_free_restores_pool(cls):
    a = cls(1024)
    off = a.alloc(32)
    a.free(off)
    stats = a.stats()
    assert stats.allocated == 0
    assert stats.free == 1024
    assert stats.num_allocations == 0


@pytest.mark.parametrize("cls", ALLOCATOR_CLASSES)
def test_distinct_allocations_dont_overlap(cls):
    a = cls(1024)
    o1 = a.alloc(64)
    o2 = a.alloc(64)
    o3 = a.alloc(64)
    offs = [o1, o2, o3]
    assert len(set(offs)) == 3
    # Each allocation occupies at least 64 bytes; none may overlap.
    spans = sorted(offs)
    for lo, hi in zip(spans, spans[1:]):
        assert hi - lo >= 64


# --------------------------------------------------------------------------
# Buddy-specific: power-of-2 rounding
# --------------------------------------------------------------------------

def test_buddy_requires_power_of_2_pool():
    with pytest.raises(ValueError):
        BuddyAllocator(1000)


def test_buddy_accepts_power_of_2_pool():
    a = BuddyAllocator(1024)
    assert a.stats().total_size == 1024


@pytest.mark.parametrize(
    "request_size,expected_block",
    [
        (1, 16),    # rounds up to the minimum block size
        (16, 16),
        (17, 32),
        (33, 64),
        (64, 64),
        (65, 128),
        (100, 128),
    ],
)
def test_buddy_rounds_up_to_power_of_2(request_size, expected_block):
    a = BuddyAllocator(1024)
    a.alloc(request_size)
    # The accounted allocation reflects the rounded-up block size.
    assert a.stats().allocated == expected_block


def test_buddy_min_block_size_is_16():
    a = BuddyAllocator(1024)
    a.alloc(1)
    assert a.stats().allocated == 16


# --------------------------------------------------------------------------
# Splitting and coalescing (both allocators)
# --------------------------------------------------------------------------

def test_buddy_split_creates_smaller_blocks():
    a = BuddyAllocator(1024)
    # A small allocation forces the 1024 block to split down repeatedly.
    a.alloc(16)
    stats = a.stats()
    # One 16-byte block allocated; remainder still available.
    assert stats.allocated == 16
    assert stats.free == 1024 - 16
    # Splitting produces multiple free blocks of varying sizes.
    assert stats.num_free_blocks >= 1


def test_buddy_coalesce_on_free():
    a = BuddyAllocator(1024)
    o1 = a.alloc(16)
    o2 = a.alloc(16)
    # Free both buddies -> they should coalesce back up.
    a.free(o1)
    a.free(o2)
    stats = a.stats()
    assert stats.allocated == 0
    assert stats.free == 1024
    # Fully coalesced back to a single free block.
    assert stats.num_free_blocks == 1


def test_buddy_recursive_coalescing():
    a = BuddyAllocator(1024)
    # Allocate enough 16-byte blocks to force several levels of splitting.
    offs = [a.alloc(16) for _ in range(8)]
    assert a.stats().allocated == 16 * 8
    # Free them all; recursive coalescing should restore one free block.
    for o in offs:
        a.free(o)
    stats = a.stats()
    assert stats.allocated == 0
    assert stats.free == 1024
    assert stats.num_free_blocks == 1


def test_buddy_buddy_address_xor():
    # Two minimum blocks split out of the same parent are buddies:
    # buddy(offset) == offset XOR block_size.
    a = BuddyAllocator(1024)
    o1 = a.alloc(16)
    o2 = a.alloc(16)
    lo, hi = sorted((o1, o2))
    assert (lo ^ 16) == hi


def test_firstfit_split_on_alloc():
    a = FirstFitAllocator(1024)
    a.alloc(100)
    stats = a.stats()
    # Exactly the requested amount is accounted as allocated.
    assert stats.allocated == 100
    assert stats.free == 1024 - 100
    # The pool was split: one allocated region + one free remainder.
    assert stats.num_free_blocks == 1


def test_firstfit_no_split_when_remainder_too_small():
    a = FirstFitAllocator(1024)
    # Allocating almost the whole pool leaves a remainder < 16 bytes,
    # so no split occurs and the whole block is handed out.
    a.alloc(1020)
    stats = a.stats()
    # The remainder (4 bytes) is too small to split, so the block is taken whole.
    assert stats.allocated == 1024
    assert stats.free == 0


def test_firstfit_coalesce_adjacent_on_free():
    a = FirstFitAllocator(1024)
    o1 = a.alloc(100)
    o2 = a.alloc(100)
    o3 = a.alloc(100)
    # Free the middle then the neighbours; adjacent frees must coalesce.
    a.free(o2)
    a.free(o1)
    a.free(o3)
    stats = a.stats()
    assert stats.allocated == 0
    assert stats.free == 1024
    # Everything coalesced back into a single free block.
    assert stats.num_free_blocks == 1


def test_firstfit_first_fit_strategy():
    a = FirstFitAllocator(1024)
    o1 = a.alloc(64)
    o2 = a.alloc(64)
    a.alloc(64)
    # Free the first two, creating a low free hole.
    a.free(o1)
    a.free(o2)
    # The next allocation that fits should land at the lowest free offset.
    o4 = a.alloc(32)
    assert o4 == o1


# --------------------------------------------------------------------------
# Allocation failure when full
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALLOCATOR_CLASSES)
def test_alloc_failure_when_full(cls):
    a = cls(1024)
    a.alloc(1024)
    with pytest.raises(AllocationError):
        a.alloc(16)


@pytest.mark.parametrize("cls", ALLOCATOR_CLASSES)
def test_alloc_larger_than_pool_fails(cls):
    a = cls(1024)
    with pytest.raises(AllocationError):
        a.alloc(2048)


@pytest.mark.parametrize("cls", ALLOCATOR_CLASSES)
def test_alloc_succeeds_again_after_free(cls):
    a = cls(1024)
    o = a.alloc(1024)
    with pytest.raises(AllocationError):
        a.alloc(16)
    a.free(o)
    # Space reclaimed -> allocation works again.
    o2 = a.alloc(512)
    assert isinstance(o2, int)


# --------------------------------------------------------------------------
# Fragmentation measurement
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALLOCATOR_CLASSES)
def test_fragmentation_zero_when_empty(cls):
    a = cls(1024)
    assert a.stats().fragmentation == 0.0


@pytest.mark.parametrize("cls", ALLOCATOR_CLASSES)
def test_fragmentation_is_ratio(cls):
    a = cls(1024)
    a.alloc(64)
    frag = a.stats().fragmentation
    assert 0.0 <= frag <= 1.0


def test_fragmentation_increases_with_holes():
    a = FirstFitAllocator(1024)
    offs = [a.alloc(64) for _ in range(8)]
    # Free alternating blocks to create many scattered free holes.
    for o in offs[::2]:
        a.free(o)
    stats = a.stats()
    # Multiple free holes -> external fragmentation should be > 0.
    assert stats.num_free_blocks > 1
    assert stats.fragmentation > 0.0


# --------------------------------------------------------------------------
# Double-free and invalid-free errors
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALLOCATOR_CLASSES)
def test_double_free_raises(cls):
    a = cls(1024)
    o = a.alloc(64)
    a.free(o)
    with pytest.raises(AllocationError):
        a.free(o)


@pytest.mark.parametrize("cls", ALLOCATOR_CLASSES)
def test_free_invalid_offset_raises(cls):
    a = cls(1024)
    a.alloc(64)
    with pytest.raises(AllocationError):
        a.free(99999)


@pytest.mark.parametrize("cls", ALLOCATOR_CLASSES)
def test_free_never_allocated_offset_raises(cls):
    a = cls(1024)
    o = a.alloc(64)
    with pytest.raises(AllocationError):
        a.free(o + 7)  # not the start of any live allocation


# --------------------------------------------------------------------------
# Stress test: random alloc/free sequences
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALLOCATOR_CLASSES)
def test_stress_random_alloc_free(cls):
    rng = random.Random(1234)
    a = cls(4096)
    live = []
    for _ in range(2000):
        if live and rng.random() < 0.5:
            off = live.pop(rng.randrange(len(live)))
            a.free(off)
        else:
            size = rng.choice([16, 32, 64, 128])
            try:
                off = a.alloc(size)
            except AllocationError:
                continue
            live.append(off)
        # Invariant: accounting stays consistent at every step.
        stats = a.stats()
        assert stats.allocated + stats.free == stats.total_size
        assert 0 <= stats.allocated <= stats.total_size
        assert 0 <= stats.free <= stats.total_size
    # Free everything that is still live; pool must return to empty.
    for off in live:
        a.free(off)
    final = a.stats()
    assert final.allocated == 0
    assert final.free == 4096


@pytest.mark.parametrize("cls", ALLOCATOR_CLASSES)
def test_stress_offsets_never_overlap(cls):
    rng = random.Random(99)
    a = cls(2048)
    live = {}  # offset -> requested size
    for _ in range(500):
        if live and rng.random() < 0.4:
            off = rng.choice(list(live))
            a.free(off)
            del live[off]
        else:
            size = rng.choice([16, 32, 64])
            try:
                off = a.alloc(size)
            except AllocationError:
                continue
            # New allocation must not overlap any live allocation's requested span.
            for o2, s2 in live.items():
                assert off + size <= o2 or o2 + s2 <= off
            live[off] = size


# --------------------------------------------------------------------------
# Comparison tooling
# --------------------------------------------------------------------------

def test_run_workload_returns_stats():
    a = FirstFitAllocator(1024)
    ops = [("alloc", 64), ("alloc", 32), ("free", 0)]
    stats = run_workload(a, ops)
    assert isinstance(stats, AllocatorStats)


def test_run_workload_alloc_and_free_by_index():
    a = FirstFitAllocator(1024)
    # ops reference earlier allocations by their order index for freeing.
    ops = [("alloc", 64), ("alloc", 64), ("free", 0), ("free", 1)]
    stats = run_workload(a, ops)
    assert stats.allocated == 0
    assert stats.free == 1024


def test_compare_returns_dict_with_both_allocators():
    ops = [("alloc", 64), ("alloc", 128), ("free", 0), ("alloc", 32)]
    result = compare(1024, ops)
    assert isinstance(result, dict)
    # Both allocators must be represented in the comparison.
    keys = {k.lower() for k in result}
    assert any("buddy" in k for k in keys)
    assert any("first" in k or "fit" in k for k in keys)


def test_compare_values_are_stats():
    ops = [("alloc", 64), ("alloc", 128), ("free", 0)]
    result = compare(1024, ops)
    for value in result.values():
        assert isinstance(value, AllocatorStats)


def test_compare_mixed_workload_consistency():
    rng = random.Random(7)
    ops = []
    n_alloc = 0
    for _ in range(40):
        if n_alloc > 0 and rng.random() < 0.4:
            ops.append(("free", rng.randrange(n_alloc)))
        else:
            ops.append(("alloc", rng.choice([16, 32, 64, 128])))
            n_alloc += 1
    result = compare(1024, ops)
    for stats in result.values():
        assert stats.allocated + stats.free == stats.total_size
        assert 0.0 <= stats.fragmentation <= 1.0
