import time
from vector_clock import VectorClock
from events import Event, EventLog


# --- VectorClock tests ---


def test_vc_increment():
    vc = VectorClock("A", 2)
    vc.increment()
    assert vc.clock["A"] == 1
    vc.increment()
    assert vc.clock["A"] == 2


def test_vc_merge():
    vc_a = VectorClock("A", 2)
    vc_b = VectorClock("B", 2)
    vc_a.increment()  # A: {A:1}
    vc_b.increment()  # B: {B:1}
    vc_b.increment()  # B: {B:2}

    vc_a.merge(vc_b)
    # After merge: element-wise max {A:1, B:2} then local increment -> {A:2, B:2}
    assert vc_a.clock["A"] == 2
    assert vc_a.clock["B"] == 2


def test_vc_happens_before():
    vc1 = VectorClock("A", 2)
    vc1.clock = {"A": 1}

    vc2 = VectorClock("A", 2)
    vc2.clock = {"A": 2, "B": 1}

    assert vc1.happens_before(vc2) is True


def test_vc_not_happens_before():
    # Concurrent VCs: neither happens-before the other
    vc1 = VectorClock("A", 2)
    vc1.clock = {"A": 2, "B": 0}

    vc2 = VectorClock("B", 2)
    vc2.clock = {"A": 0, "B": 2}

    assert vc1.happens_before(vc2) is False
    assert vc2.happens_before(vc1) is False


def test_vc_concurrent():
    vc1 = VectorClock("A", 2)
    vc1.clock = {"A": 1}

    vc2 = VectorClock("B", 2)
    vc2.clock = {"B": 1}

    assert vc1.is_concurrent(vc2) is True


def test_vc_eq():
    vc1 = VectorClock("A", 2)
    vc1.clock = {"A": 1, "B": 2}

    vc2 = VectorClock("B", 2)
    vc2.clock = {"A": 1, "B": 2}

    assert vc1 == vc2


def test_vc_le_lt():
    vc1 = VectorClock("A", 2)
    vc1.clock = {"A": 1, "B": 1}

    vc2 = VectorClock("A", 2)
    vc2.clock = {"A": 2, "B": 1}

    # vc1 <= vc2
    assert vc1 <= vc2
    # vc1 < vc2 (happens_before)
    assert vc1 < vc2
    # equal case: <= True, < False
    vc3 = VectorClock("A", 2)
    vc3.clock = {"A": 1, "B": 1}
    assert vc1 <= vc3
    assert not (vc1 < vc3)


def test_vc_copy():
    vc = VectorClock("A", 2)
    vc.increment()
    vc.increment()

    vc_copy = vc.copy()
    assert vc_copy == vc
    assert vc_copy is not vc
    assert vc_copy.clock is not vc.clock

    # Modify original, copy unchanged
    vc.increment()
    assert vc.clock["A"] == 3
    assert vc_copy.clock["A"] == 2


# --- EventLog tests ---


def test_event_log_record():
    log = EventLog()
    ts = time.time()
    for i in range(3):
        vc = VectorClock("A", 1)
        vc.clock = {"A": i + 1}
        log.record(Event(node_id="A", event_type="local", clock=vc, timestamp=ts + i))

    ordered = log.causal_order()
    assert len(ordered) == 3


def test_event_log_causal_order():
    log = EventLog()
    ts = time.time()

    # Create events in reverse causal order
    vc3 = VectorClock("A", 1)
    vc3.clock = {"A": 3}
    vc1 = VectorClock("A", 1)
    vc1.clock = {"A": 1}
    vc2 = VectorClock("A", 1)
    vc2.clock = {"A": 2}

    log.record(Event(node_id="A", event_type="local", clock=vc3, timestamp=ts + 2))
    log.record(Event(node_id="A", event_type="local", clock=vc1, timestamp=ts))
    log.record(Event(node_id="A", event_type="local", clock=vc2, timestamp=ts + 1))

    ordered = log.causal_order()
    assert ordered[0].clock.clock["A"] == 1
    assert ordered[1].clock.clock["A"] == 2
    assert ordered[2].clock.clock["A"] == 3


def test_event_log_concurrent_pairs():
    log = EventLog()
    ts = time.time()

    vc_a = VectorClock("A", 2)
    vc_a.clock = {"A": 1}
    vc_b = VectorClock("B", 2)
    vc_b.clock = {"B": 1}

    log.record(Event(node_id="A", event_type="local", clock=vc_a, timestamp=ts))
    log.record(Event(node_id="B", event_type="local", clock=vc_b, timestamp=ts + 1))

    pairs = log.concurrent_pairs()
    assert len(pairs) == 1
    assert pairs[0][0].node_id == "A"
    assert pairs[0][1].node_id == "B"


def test_event_log_find_conflicts():
    log = EventLog()
    ts = time.time()

    # Two concurrent writes to the same key
    vc_a = VectorClock("A", 2)
    vc_a.clock = {"A": 1}
    vc_b = VectorClock("B", 2)
    vc_b.clock = {"B": 1}

    log.record(Event(node_id="A", event_type="local", clock=vc_a, timestamp=ts, data={"x": 1}))
    log.record(Event(node_id="B", event_type="local", clock=vc_b, timestamp=ts + 1, data={"x": 2}))

    conflicts = log.find_conflicts("x")
    assert len(conflicts) == 1


def test_event_log_no_conflicts():
    log = EventLog()
    ts = time.time()

    # Sequential writes: A happens-before B
    vc1 = VectorClock("A", 2)
    vc1.clock = {"A": 1}
    vc2 = VectorClock("A", 2)
    vc2.clock = {"A": 2, "B": 1}

    log.record(Event(node_id="A", event_type="local", clock=vc1, timestamp=ts, data={"x": 1}))
    log.record(Event(node_id="B", event_type="local", clock=vc2, timestamp=ts + 1, data={"x": 2}))

    conflicts = log.find_conflicts("x")
    assert len(conflicts) == 0
