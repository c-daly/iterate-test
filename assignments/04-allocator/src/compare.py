"""Comparison tooling for the simulated allocators.

A *workload* is a list of operations. Each op is a tuple:

    ("alloc", size)   -> allocate ``size`` bytes
    ("free", index)   -> free the allocation made by the ``index``-th
                          successful ``alloc`` op in this workload

The ``index`` form lets the same workload be replayed against different
allocators even though the concrete offsets they hand back differ.
Failed allocations (``AllocationError``) and frees of an already-freed /
out-of-range index are skipped so a single workload can be compared
across allocators with differing capacities.
"""

from allocator import AllocationError, Allocator, AllocatorStats
from buddy import BuddyAllocator
from firstfit import FirstFitAllocator


def run_workload(allocator: Allocator, ops: list[tuple]) -> AllocatorStats:
    """Replay ``ops`` against ``allocator`` and return its final stats.

    ``free`` operations reference an earlier allocation by the index of
    the successful ``alloc`` that produced it (0-based, in op order).
    """
    # alloc_index -> offset (None once freed / if the alloc failed).
    offsets: list[int | None] = []
    for op in ops:
        kind = op[0]
        arg = op[1]
        if kind == "alloc":
            try:
                off = allocator.alloc(arg)
            except AllocationError:
                offsets.append(None)
                continue
            offsets.append(off)
        elif kind == "free":
            if 0 <= arg < len(offsets) and offsets[arg] is not None:
                try:
                    allocator.free(offsets[arg])
                except AllocationError:
                    pass
                offsets[arg] = None
        else:
            raise ValueError(f"unknown op: {kind!r}")
    return allocator.stats()


def compare(pool_size: int, ops: list[tuple]) -> dict:
    """Run the same workload on both allocators and return their stats.

    Returns a mapping of allocator name -> final ``AllocatorStats``.
    """
    results: dict[str, AllocatorStats] = {}
    results["buddy"] = run_workload(BuddyAllocator(pool_size), list(ops))
    results["firstfit"] = run_workload(FirstFitAllocator(pool_size), list(ops))
    return results
