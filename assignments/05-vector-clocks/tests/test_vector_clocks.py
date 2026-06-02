"""Tests for Assignment 5: Vector Clocks and Causal Ordering.

Drives the API described in assignments/05-vector-clocks/README.md.
"""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSIGNMENT_ROOT = os.path.dirname(_HERE)
if _ASSIGNMENT_ROOT not in sys.path:
    sys.path.insert(0, _ASSIGNMENT_ROOT)

from src.vector_clock import VectorClock  # noqa: E402
from src.events import Event, EventLog  # noqa: E402
from src.node import Node, Message  # noqa: E402
from src.simulation import Simulation  # noqa: E402


def test_init_zeroed_clock_over_all_nodes():
    vc = VectorClock("A", 3)
    assert vc.node_id == "A"
    assert all(v == 0 for v in vc.clock.values())
    assert len(vc.clock) == 3


def test_init_single_node():
    vc = VectorClock("A", 1)
    assert len(vc.clock) == 1
    assert all(v == 0 for v in vc.clock.values())


def test_increment_bumps_only_local_entry():
    vc = VectorClock("A", 3)
    bumped = vc.increment()
    assert bumped.clock["A"] == 1
    others = [v for k, v in bumped.clock.items() if k != "A"]
    assert all(v == 0 for v in others)


def test_increment_returns_vectorclock():
    vc = VectorClock("A", 2)
    assert isinstance(vc.increment(), VectorClock)


def test_increment_is_repeatable():
    vc = VectorClock("A", 2)
    vc = vc.increment().increment().increment()
    assert vc.clock["A"] == 3


def test_increment_does_not_corrupt_source_when_chained():
    vc = VectorClock("B", 2).increment().increment()
    assert vc.clock["B"] == 2
    assert vc.node_id == "B"


def test_merge_takes_elementwise_max_then_increments_local():
    a = VectorClock("A", 2).increment().increment()
    b = VectorClock("B", 2).increment()
    merged = a.merge(b)
    assert merged.clock["A"] == 3
    assert merged.clock["B"] == 1
    assert merged.node_id == "A"


def test_merge_returns_new_vectorclock():
    a = VectorClock("A", 2)
    b = VectorClock("B", 2)
    assert isinstance(a.merge(b), VectorClock)


def test_merge_with_higher_remote_keeps_higher():
    a = VectorClock("A", 2)
    b = VectorClock("B", 2).increment().increment().increment()
    merged = a.merge(b)
    assert merged.clock["B"] == 3
    assert merged.clock["A"] == 1


def test_happens_before_true_for_causal_predecessor():
    a = VectorClock("A", 2).increment()
    later = a.increment()
    assert a.happens_before(later) is True
    assert later.happens_before(a) is False


def test_happens_before_is_irreflexive():
    a = VectorClock("A", 2).increment()
    assert a.happens_before(a) is False


def test_happens_before_across_merge():
    a = VectorClock("A", 2).increment()
    b = VectorClock("B", 2).increment()
    b2 = b.merge(a)
    assert a.happens_before(b2) is True
    assert b2.happens_before(a) is False


def test_happens_before_is_transitive():
    e1 = VectorClock("A", 2).increment()
    e2 = e1.increment()
    e3 = VectorClock("B", 2).merge(e2)
    assert e1.happens_before(e2) is True
    assert e2.happens_before(e3) is True
    assert e1.happens_before(e3) is True


def test_concurrent_events_not_happens_before():
    a = VectorClock("A", 2).increment()
    b = VectorClock("B", 2).increment()
    assert a.happens_before(b) is False
    assert b.happens_before(a) is False


def test_is_concurrent_true_for_independent_events():
    a = VectorClock("A", 2).increment()
    b = VectorClock("B", 2).increment()
    assert a.is_concurrent(b) is True
    assert b.is_concurrent(a) is True


def test_is_concurrent_false_for_causal_chain():
    a = VectorClock("A", 2).increment()
    later = a.increment()
    assert a.is_concurrent(later) is False
    assert later.is_concurrent(a) is False


def test_is_concurrent_false_for_equal_clocks():
    a = VectorClock("A", 2).increment()
    b = VectorClock("A", 2).increment()
    assert a.is_concurrent(b) is False


def test_eq_for_same_counters():
    a = VectorClock("A", 2).increment()
    b = VectorClock("A", 2).increment()
    assert a == b


def test_eq_false_for_different_counters():
    a = VectorClock("A", 2).increment()
    b = VectorClock("A", 2).increment().increment()
    assert a != b


def test_le_true_when_dominated_or_equal():
    a = VectorClock("A", 2).increment()
    later = a.increment()
    assert a <= later
    assert a <= a
    assert later <= later


def test_le_false_for_concurrent():
    a = VectorClock("A", 2).increment()
    b = VectorClock("B", 2).increment()
    assert not (a <= b)
    assert not (b <= a)


def test_lt_is_strict():
    a = VectorClock("A", 2).increment()
    later = a.increment()
    assert a < later
    assert not (a < a)
    assert not (later < a)


def test_lt_matches_happens_before():
    a = VectorClock("A", 2).increment()
    b = VectorClock("B", 2).increment()
    later = a.increment()
    assert (a < later) == a.happens_before(later)
    assert (a < b) == a.happens_before(b)


def test_repr_is_string_and_mentions_node():
    vc = VectorClock("A", 2).increment()
    r = repr(vc)
    assert isinstance(r, str)
    assert "A" in r


def _ev(node_id, etype, clock, ts, data=None):
    return Event(node_id=node_id, event_type=etype, clock=clock, timestamp=ts, data=data)


def test_event_is_dataclass_with_fields():
    clk = VectorClock("A", 2).increment()
    e = _ev("A", "local", clk, 1.0, data={"k": "v"})
    assert e.node_id == "A"
    assert e.event_type == "local"
    assert e.clock is clk
    assert e.timestamp == 1.0
    assert e.data == {"k": "v"}


def test_eventlog_record_and_history():
    log = EventLog()
    clk = VectorClock("A", 2).increment()
    e = _ev("A", "local", clk, 1.0)
    log.record(e)
    assert e in log.causal_order()


def test_causal_order_respects_happens_before():
    log = EventLog()
    c1 = VectorClock("A", 2).increment()
    c2 = c1.increment()
    c3 = VectorClock("B", 2).merge(c2)
    e1 = _ev("A", "local", c1, 1.0)
    e2 = _ev("A", "local", c2, 2.0)
    e3 = _ev("B", "receive", c3, 3.0)
    log.record(e3)
    log.record(e1)
    log.record(e2)
    ordered = log.causal_order()
    assert ordered.index(e1) < ordered.index(e2)
    assert ordered.index(e2) < ordered.index(e3)


def test_concurrent_pairs_detects_independent_events():
    log = EventLog()
    a = VectorClock("A", 2).increment()
    b = VectorClock("B", 2).increment()
    ea = _ev("A", "local", a, 1.0)
    eb = _ev("B", "local", b, 1.0)
    log.record(ea)
    log.record(eb)
    pairs = log.concurrent_pairs()
    flat = {frozenset((id(x), id(y))) for x, y in pairs}
    assert frozenset((id(ea), id(eb))) in flat


def test_concurrent_pairs_empty_for_causal_chain():
    log = EventLog()
    c1 = VectorClock("A", 2).increment()
    c2 = c1.increment()
    log.record(_ev("A", "local", c1, 1.0))
    log.record(_ev("A", "local", c2, 2.0))
    assert log.concurrent_pairs() == []


def test_find_conflicts_on_concurrent_writes_same_key():
    log = EventLog()
    a = VectorClock("A", 2).increment()
    b = VectorClock("B", 2).increment()
    ea = _ev("A", "local", a, 1.0, data={"key": "x", "value": 1})
    eb = _ev("B", "local", b, 1.0, data={"key": "x", "value": 2})
    log.record(ea)
    log.record(eb)
    conflicts = log.find_conflicts("x")
    assert len(conflicts) == 1
    pair = conflicts[0]
    assert {pair[0], pair[1]} == {ea, eb}


def test_find_conflicts_none_for_causally_ordered_writes():
    log = EventLog()
    c1 = VectorClock("A", 2).increment()
    c2 = VectorClock("B", 2).merge(c1)
    e1 = _ev("A", "local", c1, 1.0, data={"key": "x", "value": 1})
    e2 = _ev("B", "local", c2, 2.0, data={"key": "x", "value": 2})
    log.record(e1)
    log.record(e2)
    assert log.find_conflicts("x") == []


def test_find_conflicts_ignores_other_keys():
    log = EventLog()
    a = VectorClock("A", 2).increment()
    b = VectorClock("B", 2).increment()
    log.record(_ev("A", "local", a, 1.0, data={"key": "x", "value": 1}))
    log.record(_ev("B", "local", b, 1.0, data={"key": "y", "value": 2}))
    assert log.find_conflicts("x") == []
    assert log.find_conflicts("y") == []


def test_node_local_event_increments_clock_and_records():
    log = EventLog()
    node = Node("A", 2, log)
    e = node.local_event(data={"hello": 1})
    assert e.node_id == "A"
    assert e.event_type == "local"
    assert e.clock.clock["A"] == 1
    assert e in log.causal_order()


def test_node_multiple_local_events_advance_clock():
    log = EventLog()
    node = Node("A", 2, log)
    node.local_event()
    e2 = node.local_event()
    assert e2.clock.clock["A"] == 2


def test_node_send_returns_event_and_message():
    log = EventLog()
    node = Node("A", 2, log)
    event, msg = node.send(data={"payload": 42})
    assert isinstance(event, Event)
    assert isinstance(msg, Message)
    assert event.event_type == "send"
    assert event.clock.clock["A"] == 1
    assert event in log.causal_order()


def test_node_receive_merges_sender_clock():
    log_a = EventLog()
    log_b = EventLog()
    a = Node("A", 2, log_a)
    b = Node("B", 2, log_b)
    a.local_event()
    send_event, msg = a.send()
    recv_event = b.receive(msg)
    assert recv_event.event_type == "receive"
    assert recv_event.clock.clock["A"] == 2
    assert recv_event.clock.clock["B"] == 1
    assert send_event.clock.happens_before(recv_event.clock)


def test_node_send_then_receive_preserves_causality():
    log_a = EventLog()
    log_b = EventLog()
    a = Node("A", 2, log_a)
    b = Node("B", 2, log_b)
    send_event, msg = a.send(data="hi")
    recv_event = b.receive(msg)
    assert send_event.clock.happens_before(recv_event.clock)
    assert not recv_event.clock.happens_before(send_event.clock)


def test_simulation_local_event():
    sim = Simulation(["A", "B"])
    e = sim.local_event("A", data={"x": 1})
    assert e.node_id == "A"
    assert e.event_type == "local"
    assert e in sim.get_history()


def test_simulation_send_and_deliver_message():
    sim = Simulation(["A", "B"])
    send_event = sim.send_message("A", "B", data={"msg": "hello"})
    assert send_event.event_type == "send"
    recv_event = sim.deliver_message("B")
    assert recv_event.event_type == "receive"
    assert send_event.clock.happens_before(recv_event.clock)


def test_simulation_get_log_and_history_consistent():
    sim = Simulation(["A", "B"])
    sim.local_event("A")
    sim.send_message("A", "B")
    sim.deliver_message("B")
    history = sim.get_history()
    log = sim.get_log()
    assert len(history) == 3
    assert set(id(e) for e in history) == set(id(e) for e in log.causal_order())


def test_simulation_deliver_without_message_raises():
    sim = Simulation(["A", "B"])
    with pytest.raises((IndexError, KeyError, RuntimeError, ValueError)):
        sim.deliver_message("B")


def test_simulation_history_in_causal_order():
    sim = Simulation(["A", "B"])
    e_send = sim.send_message("A", "B")
    e_local = sim.local_event("A")
    e_recv = sim.deliver_message("B")
    history = sim.get_history()
    assert history.index(e_send) < history.index(e_recv)
    assert e_local in history


def test_single_node_simulation_only_local_events():
    sim = Simulation(["A"])
    e1 = sim.local_event("A")
    e2 = sim.local_event("A")
    assert e1.clock.happens_before(e2.clock)
    assert sim.get_log().concurrent_pairs() == []


def test_no_messages_no_conflicts():
    sim = Simulation(["A", "B"])
    sim.local_event("A", data={"key": "k", "value": 1})
    assert sim.get_log().find_conflicts("k") == []


def test_all_concurrent_events_across_nodes():
    sim = Simulation(["A", "B", "C"])
    ea = sim.local_event("A")
    eb = sim.local_event("B")
    ec = sim.local_event("C")
    pairs = sim.get_log().concurrent_pairs()
    assert len(pairs) == 3
    assert ea.clock.is_concurrent(eb.clock)
    assert eb.clock.is_concurrent(ec.clock)
    assert ea.clock.is_concurrent(ec.clock)


def test_three_node_interleaved_communication():
    sim = Simulation(["A", "B", "C"])
    sim.local_event("A", data={"key": "x", "value": 1})
    e_send_ab = sim.send_message("A", "B")
    e_local_c = sim.local_event("C", data={"key": "x", "value": 99})
    e_recv_b = sim.deliver_message("B")
    e_send_bc = sim.send_message("B", "C")
    e_recv_c = sim.deliver_message("C")
    assert e_send_ab.clock.happens_before(e_recv_b.clock)
    assert e_recv_b.clock.happens_before(e_send_bc.clock)
    assert e_send_bc.clock.happens_before(e_recv_c.clock)
    assert e_send_ab.clock.happens_before(e_recv_c.clock)
    assert e_local_c.clock.is_concurrent(e_send_ab.clock)
    conflicts = sim.get_log().find_conflicts("x")
    assert len(conflicts) >= 1
    history = sim.get_history()
    assert history.index(e_send_ab) < history.index(e_recv_b)
    assert history.index(e_send_bc) < history.index(e_recv_c)
