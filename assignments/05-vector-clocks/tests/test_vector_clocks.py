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


# --- Node tests ---

from node import Node, Message
from simulation import Simulation


def test_node_local_event():
    log = EventLog()
    node = Node("A", 2, log)
    event = node.local_event(data="hello")
    assert node.clock.clock["A"] == 1
    assert event.event_type == "local"
    assert event.node_id == "A"
    assert event.data == "hello"
    assert len(log.causal_order()) == 1


def test_node_send():
    log = EventLog()
    node = Node("A", 2, log)
    event, msg = node.send(data="payload")
    assert event.event_type == "send"
    assert event.node_id == "A"
    assert node.clock.clock["A"] == 1
    assert isinstance(msg, Message)
    assert msg.sender_id == "A"
    assert msg.data == "payload"
    # Message clock should be a copy, not the same object
    assert msg.clock == node.clock
    assert msg.clock is not node.clock


def test_node_receive():
    log = EventLog()
    node_a = Node("A", 2, log)
    node_b = Node("B", 2, log)
    _, msg = node_a.send(data="hi")
    event = node_b.receive(msg)
    assert event.event_type == "receive"
    assert event.node_id == "B"
    # After merge: B should have A clock merged + increment
    assert node_b.clock.clock["A"] == 1
    assert node_b.clock.clock["B"] == 1


def test_simulation_basic():
    sim = Simulation(["A", "B"])
    send_evt = sim.send_message("A", "B", data="msg1")
    recv_evt = sim.deliver_message("B")
    assert send_evt.event_type == "send"
    assert recv_evt.event_type == "receive"
    history = sim.get_history()
    assert len(history) == 2
    # Send should causally precede receive
    assert send_evt.clock.happens_before(recv_evt.clock)


def test_simulation_concurrent():
    sim = Simulation(["A", "B"])
    evt_a = sim.local_event("A", data="a_work")
    evt_b = sim.local_event("B", data="b_work")
    assert evt_a.clock.is_concurrent(evt_b.clock)


def test_simulation_three_nodes():
    sim = Simulation(["A", "B", "C"])
    # A -> B
    sim.send_message("A", "B", data="ab")
    sim.deliver_message("B")
    # B -> C
    sim.send_message("B", "C", data="bc")
    sim.deliver_message("C")
    history = sim.get_history()
    assert len(history) == 4
    # A send should transitively happen-before C receive
    a_send = [e for e in history if e.node_id == "A" and e.event_type == "send"][0]
    c_recv = [e for e in history if e.node_id == "C" and e.event_type == "receive"][0]
    assert a_send.clock.happens_before(c_recv.clock)


def test_simulation_conflict_detection():
    sim = Simulation(["A", "B"])
    sim.local_event("A", data={"x": 1})
    sim.local_event("B", data={"x": 2})
    conflicts = sim.get_log().find_conflicts("x")
    assert len(conflicts) == 1


def test_simulation_no_pending():
    sim = Simulation(["A", "B"])
    try:
        sim.deliver_message("A")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_simulation_complex():
    sim = Simulation(["A", "B", "C"])
    # A does local work
    sim.local_event("A", data="a1")
    # A sends to B
    sim.send_message("A", "B", data="ab")
    # C does local work concurrently
    sim.local_event("C", data="c1")
    # B receives from A
    sim.deliver_message("B")
    # B sends to C
    sim.send_message("B", "C", data="bc")
    # C receives from B
    sim.deliver_message("C")
    # All 6 events should be in the log
    history = sim.get_history()
    assert len(history) == 6
    # Verify all node_ids present
    node_ids = {e.node_id for e in history}
    assert node_ids == {"A", "B", "C"}
