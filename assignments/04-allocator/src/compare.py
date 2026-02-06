"""Comparison tooling for allocators."""

from __future__ import annotations

from src.allocator import Allocator, AllocatorStats
from src.buddy import BuddyAllocator
from src.firstfit import FirstFitAllocator


def run_workload(
    allocator: Allocator, ops: list[tuple],
) -> AllocatorStats:
    """Run a workload of alloc/free ops and return final stats.

    ops is a list of tuples:
      ("alloc", size)  -- allocate *size* bytes
      ("free", index)  -- free the allocation returned by the *index*-th alloc
    """
    # Track allocations by their sequential index (order of alloc ops)
    alloc_results: list[int] = []

    for op in ops:
        kind = op[0]
        if kind == "alloc":
            offset = allocator.alloc(op[1])
            alloc_results.append(offset)
        elif kind == "free":
            idx = op[1]
            allocator.free(alloc_results[idx])
        else:
            msg = f"Unknown op: {kind}"
            raise ValueError(msg)

    return allocator.stats()


def compare(pool_size: int, ops: list[tuple]) -> dict:
    """Compare buddy and first-fit allocators on the same workload."""
    buddy = BuddyAllocator(pool_size)
    firstfit = FirstFitAllocator(pool_size)

    buddy_stats = run_workload(buddy, ops)
    firstfit_stats = run_workload(firstfit, ops)

    return {
        "buddy": buddy_stats,
        "firstfit": firstfit_stats,
    }
