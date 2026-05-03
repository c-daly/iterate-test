"""Workload runner and side-by-side allocator comparison."""
from __future__ import annotations

from typing import Iterable, Literal, Union

from .allocator import AllocationError, Allocator, AllocatorStats
from .buddy import BuddyAllocator
from .firstfit import FirstFitAllocator

Op = Union[tuple[Literal["alloc"], int], tuple[Literal["free"], int]]


def run_workload(allocator: Allocator, ops: Iterable[Op]) -> AllocatorStats:
    """Apply ops to allocator and return final stats.

    Allocation failures and invalid frees are tolerated so a single
    workload can be replayed against allocators with different fit
    behaviour without aborting the comparison.
    """
    for op in ops:
        kind = op[0]
        if kind == "alloc":
            size = op[1]
            try:
                allocator.alloc(size)
            except AllocationError:
                pass
        elif kind == "free":
            offset = op[1]
            try:
                allocator.free(offset)
            except AllocationError:
                pass
        else:
            raise ValueError(f"unknown op: {kind!r}")
    return allocator.stats()


def compare(pool_size: int, ops: list[Op]) -> dict[str, AllocatorStats]:
    """Run the same workload against buddy and first-fit allocators."""
    return {
        "buddy": run_workload(BuddyAllocator(pool_size), ops),
        "firstfit": run_workload(FirstFitAllocator(pool_size), ops),
    }
