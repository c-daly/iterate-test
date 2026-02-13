from abc import ABC, abstractmethod
from dataclasses import dataclass


class AllocationError(Exception):
    pass


@dataclass
class AllocatorStats:
    total_size: int
    allocated: int
    free: int
    num_allocations: int
    num_free_blocks: int
    fragmentation: float  # 1 - (largest_free_block / total_free). 0 if no free or one block.


class Allocator(ABC):
    def __init__(self, pool_size: int):
        self.pool_size = pool_size

    @abstractmethod
    def alloc(self, size: int) -> int:
        ...

    @abstractmethod
    def free(self, offset: int) -> None:
        ...

    @abstractmethod
    def stats(self) -> AllocatorStats:
        ...

    def dump(self) -> str:
        return str(self.stats())
