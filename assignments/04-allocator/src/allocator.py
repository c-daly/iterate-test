"""Base abstractions for the simulated memory allocators.

No real memory is ever touched. Allocators model a pool of ``pool_size``
bytes purely through offset bookkeeping; ``alloc`` returns an integer
offset into the simulated pool.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class AllocationError(Exception):
    """Raised on a failed allocation or an invalid/double free."""


@dataclass
class AllocatorStats:
    """Snapshot of an allocator's state.

    Attributes:
        total_size: Total size of the simulated pool, in bytes.
        allocated: Bytes currently handed out to live allocations.
        free: Bytes currently available (``total_size - allocated``).
        num_allocations: Count of live allocations.
        num_free_blocks: Count of distinct free regions.
        fragmentation: External fragmentation ratio in ``[0.0, 1.0]``.
    """

    total_size: int
    allocated: int
    free: int
    num_allocations: int
    num_free_blocks: int
    fragmentation: float  # external fragmentation ratio


class Allocator(ABC):
    """Abstract base for a simulated memory allocator."""

    def __init__(self, pool_size: int):
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        self.pool_size = pool_size

    @abstractmethod
    def alloc(self, size: int) -> int:
        """Allocate ``size`` bytes, returning the offset of the region.

        Raises:
            AllocationError: If the request cannot be satisfied.
        """
        raise NotImplementedError

    @abstractmethod
    def free(self, offset: int) -> None:
        """Free the allocation that starts at ``offset``.

        Raises:
            AllocationError: If ``offset`` is not the start of a live
                allocation (invalid or already freed).
        """
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> AllocatorStats:
        """Return a snapshot of the allocator state."""
        raise NotImplementedError

    def dump(self) -> str:
        """Return a human-readable memory map.

        The default rendering summarises the current stats; subclasses may
        override to show the full block layout.
        """
        s = self.stats()
        return (
            f"{type(self).__name__}(total={s.total_size}, "
            f"allocated={s.allocated}, free={s.free}, "
            f"allocations={s.num_allocations}, "
            f"free_blocks={s.num_free_blocks}, "
            f"fragmentation={s.fragmentation:.3f})"
        )
