# Assignment 5: Vector Clocks and Causal Ordering

## Overview

Implement vector clocks for tracking causality in a distributed system simulation. Support event logging, causal ordering, and conflict detection.

## Requirements

### Vector Clock
File: `src/vector_clock.py`

```python
class VectorClock:
    def __init__(self, node_id: str, num_nodes: int): ...
    def increment(self) -> "VectorClock": ...
    def merge(self, other: "VectorClock") -> "VectorClock": ...
    def happens_before(self, other: "VectorClock") -> bool: ...
    def is_concurrent(self, other: "VectorClock") -> bool: ...
    def __eq__(self, other): ...
    def __le__(self, other): ...
    def __lt__(self, other): ...
    def __repr__(self) -> str: ...
```

### Event Log
File: `src/events.py`

```python
@dataclass
class Event:
    node_id: str
    event_type: str  # "local", "send", "receive"
    clock: VectorClock
    timestamp: float
    data: Any = None

class EventLog:
    def record(self, event: Event) -> None: ...
    def causal_order(self) -> list[Event]: ...
    def concurrent_pairs(self) -> list[tuple[Event, Event]]: ...
    def find_conflicts(self, key: str) -> list[tuple[Event, Event]]: ...
```

### Node
File: `src/node.py`

```python
class Node:
    def __init__(self, node_id: str, num_nodes: int, event_log: EventLog): ...
    def local_event(self, data: Any = None) -> Event: ...
    def send(self, data: Any = None) -> tuple[Event, "Message"]: ...
    def receive(self, message: "Message") -> Event: ...
```

### Simulation
File: `src/simulation.py`

```python
class Simulation:
    def __init__(self, node_ids: list[str]): ...
    def local_event(self, node_id: str, data: Any = None) -> Event: ...
    def send_message(self, from_id: str, to_id: str, data: Any = None) -> Event: ...
    def deliver_message(self, to_id: str) -> Event: ...
    def get_log(self) -> EventLog: ...
    def get_history(self) -> list[Event]: ...
```

## Constraints

- Vector clocks use dict mapping node_id -> counter.
- `happens_before` is strict partial order (irreflexive, transitive).
- `concurrent` means neither happens-before the other.
- `merge` takes element-wise max and increments local counter.
- Thread safety NOT required.

## Test Expectations

Tests in `tests/test_vector_clocks.py` should cover:
- VectorClock: increment, merge, happens_before, concurrent, comparison ops
- EventLog: recording, causal ordering, concurrent pair detection
- Node: local events, send/receive message passing
- Simulation: multi-node scenarios with message delivery
- Conflict detection on concurrent writes to same key
- Edge cases: single node, no messages, all-concurrent events
- Complex scenario: 3+ nodes with interleaved communication
