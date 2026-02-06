# Assignment 4: Memory Allocator

## Overview

Implement two memory allocation strategies — buddy allocator and first-fit allocator — operating on a simulated memory pool. Include comparison tooling.

## Requirements

### Base Class
File: `src/allocator.py`

```python
class AllocationError(Exception): ...

@dataclass
class AllocatorStats:
    total_size: int
    allocated: int
    free: int
    num_allocations: int
    num_free_blocks: int
    fragmentation: float  # external fragmentation ratio

class Allocator(ABC):
    def __init__(self, pool_size: int): ...
    @abstractmethod
    def alloc(self, size: int) -> int: ...  # returns offset
    @abstractmethod
    def free(self, offset: int) -> None: ...
    @abstractmethod
    def stats(self) -> AllocatorStats: ...
    def dump(self) -> str: ...  # human-readable memory map
```

### Buddy Allocator
File: `src/buddy.py`

- Power-of-2 block sizes only (round up requests).
- Split larger blocks when needed.
- Coalesce buddy pairs on free (recursively).
- Buddy address = offset XOR block_size.

### First-Fit Allocator
File: `src/firstfit.py`

- Maintain sorted free list.
- First-fit search strategy.
- Split blocks if remainder is large enough (>= 16 bytes).
- Coalesce adjacent free blocks on free.

### Comparison
File: `src/compare.py`

```python
def run_workload(allocator: Allocator, ops: list[tuple]) -> AllocatorStats: ...
def compare(pool_size: int, ops: list[tuple]) -> dict: ...
```

## Constraints

- Pool size must be power of 2 for buddy allocator.
- `alloc` raises `AllocationError` on failure.
- `free` on invalid/already-freed offset raises `AllocationError`.
- No real memory — simulate with offset tracking.

## Test Expectations

Tests in `tests/test_allocator.py` should cover:
- Single alloc/free for both allocators
- Power-of-2 rounding (buddy)
- Block splitting and coalescing (both)
- Recursive buddy coalescing
- Allocation failure when full
- Fragmentation measurement
- Double-free and invalid-free errors
- Stress test: random alloc/free sequences
- Comparison function with mixed workloads
