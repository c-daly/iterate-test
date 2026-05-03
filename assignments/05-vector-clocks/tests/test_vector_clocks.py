"""Tests for vector clocks, events, nodes, and simulation."""
from __future__ import annotations

import time

import pytest

from src.events import Event, EventLog
from src.node import Message, Node
from src.simulation import Simulation
from src.vector_clock import VectorClock


class TestVectorClock:
    def test_init_zero(self):
        vc = VectorClock("A", 3)
        assert vc.get("A") == 0
        assert vc.get("B") == 0
        assert vc.get("C") == 0

    def test_increment_returns_new_clock(self):
        vc = VectorClock("A", 2)
        vc2 = vc.increment()
        assert vc.get("A") == 0
        assert vc2.get("A") == 1
        assert vc is not vc2

    def test_increment_only_local(self):
        vc = VectorClock("A", 2).increment().increment()
        assert vc.get("A") == 2
        assert vc.get("B") == 0

    def test_merge_takes_elementwise_max_then_increments_local(self):
        a = VectorClock("A", 2).increment().increment()
        b = VectorClock("B", 2).increment().increment().increment()
        merged = a.merge(b)
        assert merged.get("A") == 3
        assert merged.get("B") == 3

    def test_merge_returns_new_clock(self):
        a = VectorClock("A", 2)
        b = VectorClock("B", 2)
        merged = a.merge(b)
        assert merged is not a
        assert merged is not b
        assert a.get("A") == 0
        assert b.get("B") == 0

    def test_happens_before_basic(self):
        a = VectorClock("A", 2)
        b = a.increment()
        assert a.happens_before(b)
        assert not b.happens_before(a)

    def test_happens_before_irreflexive(self):
        a = VectorClock("A", 2).increment()
        assert not a.happens_before(a)

    def test_happens_before_transitive(self):
        a = VectorClock("A", 2)
        b = a.increment()
        c = b.increment()
        assert a.happens_before(b)
        assert b.happens_before(c)
        assert a.happens_before(c)

    def test_concurrent_when_neither_precedes(self):
        a = VectorClock("A", 2).increment()
        b = VectorClock("B", 2).increment()
        assert a.is_concurrent(b)
        assert b.is_concurrent(a)
        assert not a.happens_before(b)
        assert not b.happens_before(a)

    def test_concurrent_is_symmetric(self):
        a = VectorClock("A", 3).increment()
        b = VectorClock("B", 3).increment()
        assert a.is_concurrent(b) == b.is_concurrent(a)

    def test_equal_clocks_not_concurrent_not_hb(self):
        a = VectorClock("A", 2).increment()
        b = VectorClock("A", 2).increment()
        assert a == b
        assert not a.happens_before(b)
        assert not a.is_concurrent(b)

    def test_eq_and_le_lt(self):
        a = VectorClock("A", 2)
        b = a.increment()
        assert a == VectorClock("A", 2)
        assert a <= b
        assert a < b
        assert not (b <= a)
        assert not (b < a)
        assert a <= a
        assert not (a < a)

    def test_eq_returns_notimplemented_for_unrelated(self):
        a = VectorClock("A", 2)
        assert a != 42
        assert a != "foo"

    def test_repr_contains_class_name(self):
        a = VectorClock("A", 2)
        assert "VectorClock" in repr(a)

    def test_after_merge_sender_hb_receiver(self):
        a = VectorClock("A", 2).increment()
        b = VectorClock("B", 2).merge(a)
        assert a.happens_before(b)


class TestEventLog:
    def test_record_appends(self):
        log = EventLog()
        ev = Event("A", "local", VectorClock("A", 1).increment(), 1.0)
        log.record(ev)
        assert len(log.events) == 1
        assert log.events[0] is ev

    def test_causal_order_respects_happens_before(self):
        log = EventLog()
        c1 = VectorClock("A", 2).increment()
        c2 = c1.increment()
        e2 = Event("A", "local", c2, 2.0)
        e1 = Event("A", "local", c1, 1.0)
        log.record(e2)
        log.record(e1)
        ordered = log.causal_order()
        assert ordered.index(e1) < ordered.index(e2)

    def test_causal_order_when_wall_time_disagrees(self):
        log = EventLog()
        c1 = VectorClock("A", 2).increment()
        c2 = c1.increment()
        e1 = Event("A", "local", c1, 100.0)
        e2 = Event("A", "local", c2, 50.0)
        log.record(e1)
        log.record(e2)
        ordered = log.causal_order()
        assert ordered.index(e1) < ordered.index(e2)

    def test_concurrent_pairs(self):
        log = EventLog()
        a = VectorClock("A", 2).increment()
        b = VectorClock("B", 2).increment()
        ea = Event("A", "local", a, 1.0)
        eb = Event("B", "local", b, 1.1)
        log.record(ea)
        log.record(eb)
        pairs = log.concurrent_pairs()
        assert len(pairs) == 1
        pair_set = {id(pairs[0][0]), id(pairs[0][1])}
        assert pair_set == {id(ea), id(eb)}

    def test_no_concurrent_pairs_when_causally_ordered(self):
        log = EventLog()
        c1 = VectorClock("A", 1).increment()
        c2 = c1.increment()
        e1 = Event("A", "local", c1, 1.0)
        e2 = Event("A", "local", c2, 2.0)
        log.record(e1)
        log.record(e2)
        assert log.concurrent_pairs() == []

    def test_find_conflicts_same_key_concurrent(self):
        log = EventLog()
        a = VectorClock("A", 2).increment()
        b = VectorClock("B", 2).increment()
        ea = Event("A", "local", a, 1.0, data={"key": "x", "value": 1})
        eb = Event("B", "local", b, 1.1, data={"key": "x", "value": 2})
        log.record(ea)
        log.record(eb)
        conflicts = log.find_conflicts("x")
        assert len(conflicts) == 1

    def test_find_conflicts_different_keys_no_conflict(self):
        log = EventLog()
        a = VectorClock("A", 2).increment()
        b = VectorClock("B", 2).increment()
        ea = Event("A", "local", a, 1.0, data={"key": "x", "value": 1})
        eb = Event("B", "local", b, 1.1, data={"key": "y", "value": 2})
        log.record(ea)
        log.record(eb)
        assert log.find_conflicts("x") == []
        assert log.find_conflicts("y") == []

    def test_find_conflicts_causally_ordered_no_conflict(self):
        log = EventLog()
        c1 = VectorClock("A", 1).increment()
        c2 = c1.increment()
        e1 = Event("A", "local", c1, 1.0, data={"key": "x", "value": 1})
        e2 = Event("A", "local", c2, 2.0, data={"key": "x", "value": 2})
        log.record(e1)
        log.record(e2)
        assert log.find_conflicts("x") == []


class TestNode:
    def test_local_event_increments_clock(self):
        log = EventLog()
        n = Node("A", 2, log)
        ev = n.local_event(data={"k": 1})
        assert ev.event_type == "local"
        assert ev.node_id == "A"
        assert ev.clock.get("A") == 1
        assert ev in log.events

    def test_send_increments_clock_and_returns_message(self):
        log = EventLog()
        n = Node("A", 2, log)
        ev, msg = n.send(data="hi")
        assert ev.event_type == "send"
        assert ev.clock.get("A") == 1
        assert isinstance(msg, Message)
        assert msg.sender == "A"
        assert msg.data == "hi"
        assert msg.clock == ev.clock

    def test_receive_merges_clock(self):
        log_a = EventLog()
        log_b = EventLog()
        a = Node("A", 2, log_a)
        b = Node("B", 2, log_b)
        _, msg = a.send(data="ping")
        rev = b.receive(msg)
        assert rev.event_type == "receive"
        assert rev.clock.get("A") == 1
        assert rev.clock.get("B") == 1

    def test_send_then_receive_establishes_happens_before(self):
        log = EventLog()
        a = Node("A", 2, log)
        b = Node("B", 2, log)
        send_ev, msg = a.send(data="x")
        recv_ev = b.receive(msg)
        assert send_ev.clock.happens_before(recv_ev.clock)


class TestSimulation:
    def test_single_node_local_events(self):
        sim = Simulation(["A"])
        e1 = sim.local_event("A", data=1)
        e2 = sim.local_event("A", data=2)
        assert e1.clock.happens_before(e2.clock)
        assert sim.get_log().concurrent_pairs() == []

    def test_two_nodes_send_receive(self):
        sim = Simulation(["A", "B"])
        send_ev = sim.send_message("A", "B", data="hi")
        recv_ev = sim.deliver_message("B")
        assert send_ev.clock.happens_before(recv_ev.clock)

    def test_all_concurrent_no_communication(self):
        sim = Simulation(["A", "B", "C"])
        ea = sim.local_event("A")
        eb = sim.local_event("B")
        ec = sim.local_event("C")
        pairs = sim.get_log().concurrent_pairs()
        assert len(pairs) == 3
        for x, y in [(ea, eb), (ea, ec), (eb, ec)]:
            assert x.clock.is_concurrent(y.clock)

    def test_three_node_complex_scenario(self):
        sim = Simulation(["A", "B", "C"])
        a1 = sim.local_event("A", data="a-local")
        a_send = sim.send_message("A", "B", data="a->b")
        b_recv = sim.deliver_message("B")
        b_send = sim.send_message("B", "C", data="b->c")
        c_recv = sim.deliver_message("C")
        assert a1.clock.happens_before(a_send.clock)
        assert a_send.clock.happens_before(b_recv.clock)
        assert b_recv.clock.happens_before(b_send.clock)
        assert b_send.clock.happens_before(c_recv.clock)
        assert a1.clock.happens_before(c_recv.clock)

    def test_get_history_returns_causal_order(self):
        sim = Simulation(["A", "B"])
        a = sim.local_event("A")
        sim.send_message("A", "B")
        b_recv = sim.deliver_message("B")
        history = sim.get_history()
        assert history.index(a) < history.index(b_recv)

    def test_deliver_with_no_pending_raises(self):
        sim = Simulation(["A", "B"])
        with pytest.raises((IndexError, KeyError, ValueError)):
            sim.deliver_message("B")

    def test_conflict_detection_concurrent_writes_same_key(self):
        sim = Simulation(["A", "B"])
        sim.local_event("A", data={"key": "x", "value": 1})
        sim.local_event("B", data={"key": "x", "value": 2})
        conflicts = sim.get_log().find_conflicts("x")
        assert len(conflicts) == 1

    def test_no_conflict_after_synchronization(self):
        sim = Simulation(["A", "B"])
        sim.local_event("A", data={"key": "x", "value": 1})
        sim.send_message("A", "B", data={"key": "x", "value": 1})
        sim.deliver_message("B")
        sim.local_event("B", data={"key": "x", "value": 2})
        conflicts = sim.get_log().find_conflicts("x")
        assert conflicts == []

    def test_multiple_pending_messages_fifo(self):
        sim = Simulation(["A", "B"])
        sim.send_message("A", "B", data="first")
        sim.send_message("A", "B", data="second")
        first = sim.deliver_message("B")
        second = sim.deliver_message("B")
        assert first.data == "first"
        assert second.data == "second"

    def test_timestamps_are_floats(self):
        sim = Simulation(["A"])
        before = time.time()
        ev = sim.local_event("A")
        after = time.time()
        assert isinstance(ev.timestamp, float)
        assert before <= ev.timestamp <= after
