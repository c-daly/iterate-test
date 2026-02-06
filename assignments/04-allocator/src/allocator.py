"""Base allocator classes and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class AllocationError(Exception):
    """Raised when allocation or free fails."""


@dataclass
class AllocatorStats:
    total_size: int
    allocated: int
    free: int
    num_allocations: int
    num_free_blocks: int
    fragmentation: float  # external fragmentation ratio


class Allocator(ABC):
    def __init__(self, pool_size: int) -> None:
        self.pool_size = pool_size

    @abstractmethod
    def alloc(self, size: int) -> int:
        """Allocate size bytes, return offset."""
        ...

    @abstractmethod
    def free(self, offset: int) -> None:
        """Free block at offset."""
        ...

    @abstractmethod
    def stats(self) -> AllocatorStats:
        """Return current allocator statistics."""
        ...

    def dump(self) -> str:
        """Return human-readable memory map."""
        raise NotImplementedError
