"""Workload runner and side-by-side comparison harness for allocators.

A workload is a list of operation tuples:

* ``("alloc", size)``  — allocate ``size`` bytes; the returned offset is
  recorded against the index of *this* op so subsequent ``free`` ops can
  refer to it.
* ``("free", op_index)`` — free the offset that was returned by the
  ``alloc`` op at ``op_index``.

Allocation failures during a workload are silently skipped so a single
workload can drive both allocators end-to-end without aborting.
"""
from __future__ import annotations

from typing import Sequence

from .allocator import AllocationError, Allocator, AllocatorStats
from .buddy import BuddyAllocator
from .firstfit import FirstFitAllocator


def run_workload(allocator: Allocator, ops: Sequence[tuple]) -> AllocatorStats:
    """Execute ``ops`` against ``allocator`` and return its final stats.

    Args:
        allocator: An :class:`Allocator` instance to drive.
        ops: Sequence of ``("alloc", size)`` / ``("free", op_index)`` tuples.

    Returns:
        The allocator stats snapshot after the last op completes.

    Raises:
        ValueError: if an op tuple has an unrecognised verb.
    """
    offsets: dict[int, int] = {}
    for i, op in enumerate(ops):
        verb = op[0]
        if verb == "alloc":
            size = int(op[1])
            try:
                offsets[i] = allocator.alloc(size)
            except AllocationError:
                # Skip impossible allocations so the workload can continue.
                continue
        elif verb == "free":
            target = int(op[1])
            if target in offsets:
                try:
                    allocator.free(offsets.pop(target))
                except AllocationError:
                    continue
        else:
            raise ValueError(f"unknown op verb: {verb!r}")
    return allocator.stats()


def compare(pool_size: int, ops: Sequence[tuple]) -> dict[str, AllocatorStats]:
    """Run the same workload against buddy and first-fit, return both stats.

    Args:
        pool_size: Pool size for each allocator. Must be a power of two so the
            buddy allocator accepts it.
        ops: Workload operations (see :func:`run_workload`).

    Returns:
        A dict ``{"buddy": stats, "firstfit": stats}``.
    """
    return {
        "buddy": run_workload(BuddyAllocator(pool_size), ops),
        "firstfit": run_workload(FirstFitAllocator(pool_size), ops),
    }
