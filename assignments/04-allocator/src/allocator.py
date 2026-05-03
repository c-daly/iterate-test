"""Base allocator interface and shared dataclasses.

Defines :class:`AllocationError`, the :class:`AllocatorStats` dataclass
returned by every allocator, and the abstract :class:`Allocator` base class
that both buddy and first-fit implementations inherit from.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class AllocationError(Exception):
    """Raised on allocation failure, double free, or invalid free."""


@dataclass
class AllocatorStats:
    """Snapshot of allocator state.

    Attributes:
        total_size: Pool capacity in bytes.
        allocated: Total bytes currently allocated to live blocks.
        free: Total bytes currently free across all free blocks.
        num_allocations: Number of currently outstanding allocations.
        num_free_blocks: Number of distinct free blocks (segments).
        fragmentation: External fragmentation ratio in ``[0.0, 1.0]``.
            Defined as ``1 - (largest_free / total_free)`` when there is
            free space, else ``0.0``.
    """

    total_size: int
    allocated: int
    free: int
    num_allocations: int
    num_free_blocks: int
    fragmentation: float


class Allocator(ABC):
    """Abstract base class for memory allocators operating on a simulated pool."""

    def __init__(self, pool_size: int) -> None:
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        self.pool_size = pool_size

    @abstractmethod
    def alloc(self, size: int) -> int:
        """Allocate ``size`` bytes; return offset into the pool.

        Raises:
            AllocationError: if no suitable block is available or ``size`` is invalid.
        """

    @abstractmethod
    def free(self, offset: int) -> None:
        """Free the allocation that starts at ``offset``.

        Raises:
            AllocationError: if ``offset`` is not a live allocation.
        """

    @abstractmethod
    def stats(self) -> AllocatorStats:
        """Return a snapshot of current allocator statistics."""

    def dump(self) -> str:
        """Return a human-readable textual map of the pool.

        Default implementation just summarises stats; subclasses are
        encouraged to override with a richer block-by-block layout.
        """
        s = self.stats()
        return (
            f"Allocator(total={s.total_size}, allocated={s.allocated}, "
            f"free={s.free}, blocks={s.num_free_blocks}, "
            f"fragmentation={s.fragmentation:.3f})"
        )
