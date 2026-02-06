# Assignment 1: Probabilistic Skip List

## Overview

Implement a probabilistic skip list — a layered linked-list data structure that provides O(log n) expected time for search, insert, and delete operations using randomized leveling.

## Requirements

### Data Structure: `SkipList`

File: `src/skiplist.py`

```python
class SkipList:
    def __init__(self, max_level: int = 16, p: float = 0.5): ...
    def insert(self, key: int, value: Any) -> None: ...
    def delete(self, key: int) -> bool: ...
    def search(self, key: int) -> Any | None: ...
    def range_query(self, lo: int, hi: int) -> list[tuple[int, Any]]: ...
    def __len__(self) -> int: ...
    def __contains__(self, key: int) -> bool: ...
    def __iter__(self) -> Iterator[tuple[int, Any]]: ...
    def __repr__(self) -> str: ...
```

### Behavior

- **insert(key, value)**: Insert key-value pair. If key exists, update value. Promote to random level using coin-flip probability `p`.
- **delete(key)**: Remove key from all levels. Return True if found, False otherwise.
- **search(key)**: Return associated value or None.
- **range_query(lo, hi)**: Return sorted list of (key, value) tuples where lo <= key <= hi.
- **__len__**: Return number of elements.
- **__contains__**: Return True if key exists.
- **__iter__**: Yield (key, value) tuples in sorted order.

### Constraints

- Keys are integers, values are any type.
- Level generation uses geometric distribution with parameter `p`.
- Max level caps the number of layers.
- Thread safety is NOT required.

## Test Expectations

Tests should go in `tests/test_skiplist.py` and cover:
- Basic CRUD operations
- Duplicate key handling (update semantics)
- Range queries (inclusive bounds, empty ranges)
- Edge cases: empty list operations, single element, delete nonexistent
- Probabilistic behavior: large insertions maintain O(log n) height
- Iterator correctness and ordering
- Stress test: 1000+ insertions with correctness verification
