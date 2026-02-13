from allocator import Allocator, AllocatorStats
from buddy import BuddyAllocator
from firstfit import FirstFitAllocator


def run_workload(allocator: Allocator, ops: list[tuple]) -> AllocatorStats:
    """Execute a workload on an allocator and return final stats.

    ops: list of tuples:
        ("alloc", size) -> allocate, store returned offset
        ("free", index) -> free the offset returned by the index-th alloc op
    """
    alloc_results = {}
    alloc_idx = 0
    for op in ops:
        if op[0] == "alloc":
            offset = allocator.alloc(op[1])
            alloc_results[alloc_idx] = offset
            alloc_idx += 1
        elif op[0] == "free":
            idx = op[1]
            if idx in alloc_results:
                allocator.free(alloc_results[idx])
                del alloc_results[idx]
    return allocator.stats()


def compare(pool_size: int, ops: list[tuple]) -> dict:
    """Run same workload on both allocators, return dict with buddy and firstfit keys."""
    buddy = BuddyAllocator(pool_size)
    ff = FirstFitAllocator(pool_size)
    return {
        "buddy": run_workload(buddy, ops),
        "firstfit": run_workload(ff, ops),
    }
