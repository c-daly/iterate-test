"""Base allocator interface and shared types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class AllocationError(Exception):
    """Raised on allocation failure or invalid free."""


@dataclass
class AllocatorStats:
    total_size: int
    allocated: int
    free: int
    num_allocations: int
    num_free_blocks: int
    fragmentation: float  # external fragmentation ratio in [0.0, 1.0]


class Allocator(ABC):
    """Abstract memory allocator over a simulated pool.

    Subclasses track offsets only; no real memory is reserved.
    """

    def __init__(self, pool_size: int) -> None:
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        self.pool_size = pool_size

    @abstractmethod
    def alloc(self, size: int) -> int:
        """Allocate `size` bytes; return the offset.

        Raises AllocationError if the request cannot be satisfied.
        """

    @abstractmethod
    def free(self, offset: int) -> None:
        """Release the block previously returned by alloc().

        Raises AllocationError on invalid or already-freed offsets.
        """

    @abstractmethod
    def stats(self) -> AllocatorStats:
        """Return current allocator statistics."""

    def dump(self) -> str:
        """Return a human-readable map of the pool.

        Default uses stats(); subclasses may override with richer output.
        """
        s = self.stats()
        return (
            f"{type(self).__name__}(pool={s.total_size}) "
            f"allocated={s.allocated} free={s.free} "
            f"allocs={s.num_allocations} free_blocks={s.num_free_blocks} "
            f"frag={s.fragmentation:.3f}"
        )


def compute_fragmentation(free_blocks: list[int], total_free: int) -> float:
    """External fragmentation: 1 - (largest_free / total_free).

    Returns 0.0 when there is no free space.
    """
    if total_free <= 0 or not free_blocks:
        return 0.0
    return 1.0 - (max(free_blocks) / total_free)
