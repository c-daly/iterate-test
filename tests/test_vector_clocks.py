"""Tests for vector clocks, events, nodes, and simulation."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.events import Event, EventLog  # noqa: E402
from src.node import Message, Node  # noqa: E402
from src.simulation import Simulation  # noqa: E402
from src.vector_clock import VectorClock  # noqa: E402


# --- VectorClock ---------------------------------------------------------

class TestVectorClock:
    def test_init_zeros(self):
        vc = VectorClock("A", 3)
        assert all(v == 0 for v in vc.counters.values())
        assert len(vc.counters) == 3
        assert vc.node_id == "A"
        assert "A" in vc.counters

    def test_increment_returns_new_instance(self):
        vc = VectorClock("A", 2)
        original_snapshot = dict(vc.counters)
        vc2 = vc.increment()
        assert vc.counters == original_snapshot
        assert vc2 is not vc
        assert vc2.counters["A"] == 1

    def test_increment_multiple_times(self):
        vc = VectorClock("A", 2)
        vc = vc.increment().increment().increment()
        assert vc.counters["A"] == 3

    def test_merge_takes_elementwise_max_and_increments_local(self):
        a = VectorClock("A", 2)
        b = VectorClock("B", 2)
        a = a.increment().increment()
        b = b.increment()
        merged = a.merge(b)
        assert merged.counters["A"] == 3
        assert merged.counters["B"] == 1
        assert a.counters["A"] == 2
        assert a.counters.get("B", 0) == 0

    def test_happens_before_strict_irreflexive(self):
        a = VectorClock("A", 2)
        a1 = a.increment()
        a2 = a1.increment()
        assert a1.happens_before(a2)
        assert not a2.happens_before(a1)
        assert not a1.happens_before(a1)

    def test_happens_before_transitive(self):
        a0 = VectorClock("A", 2)
        a1 = a0.increment()
        a2 = a1.increment()
        a3 = a2.increment()
        assert a1.happens_before(a2)
        assert a2.happens_before(a3)
        assert a1.happens_before(a3)

    def test_concurrent_basic(self):
        a = VectorClock("A", 2).increment()
        b = VectorClock("B", 2).increment()
        assert a.is_concurrent(b)
        assert b.is_concurrent(a)
        assert not a.happens_before(b)
        assert not b.happens_before(a)

    def test_concurrent_false_for_equal_clocks(self):
        a = VectorClock("A", 2).increment()
        b = VectorClock("A", 2).increment()
        assert a == b
        assert not a.is_concurrent(b)

    def test_concurrent_false_for_ordered(self):
        a = VectorClock("A", 2)
        a1 = a.increment()
        a2 = a1.increment()
        assert not a1.is_concurrent(a2)
        assert not a2.is_concurrent(a1)

    def test_equality_basic_and_typed(self):
        a = VectorClock("A", 2).increment()
        b = VectorClock("A", 2).increment()
        c = VectorClock("A", 2).increment().increment()
        assert a == b
        assert a != c
        assert (a == "not a clock") is False

    def test_le_and_lt(self):
        a = VectorClock("A", 2)
        a1 = a.increment()
        a2 = a1.increment()
        assert a1 <= a2
        assert a1 < a2
        assert a1 <= a1
        assert not (a1 < a1)

    def test_le_with_concurrent_is_false(self):
        a = VectorClock("A", 2).increment()
        b = VectorClock("B", 2).increment()
        assert not (a <= b)
        assert not (b <= a)

    def test_repr_contains_node_and_label(self):
        vc = VectorClock("A", 2).increment()
        r = repr(vc)
        assert "A" in r
        assert "VectorClock" in r

    def test_single_node_clock(self):
        vc = VectorClock("only", 1)
        assert len(vc.counters) == 1
        vc2 = vc.increment()
        assert vc.happens_before(vc2)
        assert not vc2.is_concurrent(vc)


# --- EventLog ------------------------------------------------------------

class TestEventLog:
    def _ev(self, node_id, etype, clock, ts, data=None):
        return Event(node_id=node_id, event_type=etype, clock=clock, timestamp=ts, data=data)

    def test_record_appends(self):
        log = EventLog()
        a = VectorClock("A", 2).increment()
        ev = self._ev("A", "local", a, 0.0)
        log.record(ev)
        assert ev in log.events
        assert len(log.events) == 1

    def test_causal_order_chain(self):
        log = EventLog()
        a0 = VectorClock("A", 2)
        a1 = a0.increment()
        a2 = a1.increment()
        a3 = a2.increment()
        e3 = self._ev("A", "local", a3, 3.0)
        e1 = self._ev("A", "local", a1, 1.0)
        e2 = self._ev("A", "local", a2, 2.0)
        log.record(e3)
        log.record(e1)
        log.record(e2)
        ordered = log.causal_order()
        assert ordered == [e1, e2, e3]

    def test_causal_order_concurrent_tiebreak(self):
        log = EventLog()
        a = VectorClock("A", 2).increment()
        b = VectorClock("B", 2).increment()
        ea = self._ev("A", "local", a, 1.0)
        eb = self._ev("B", "local", b, 2.0)
        log.record(eb)
        log.record(ea)
        ordered = log.causal_order()
        assert ordered.index(ea) < ordered.index(eb)

    def test_concurrent_pairs(self):
        log = EventLog()
        a = VectorClock("A", 2).increment()
        b = VectorClock("B", 2).increment()
        ea = self._ev("A", "local", a, 1.0)
        eb = self._ev("B", "local", b, 2.0)
        log.record(ea)
        log.record(eb)
        pairs = log.concurrent_pairs()
        assert len(pairs) == 1
        pair = pairs[0]
        assert {id(pair[0]), id(pair[1])} == {id(ea), id(eb)}

    def test_concurrent_pairs_empty_for_chain(self):
        log = EventLog()
        a0 = VectorClock("A", 2)
        a1 = a0.increment()
        a2 = a1.increment()
        log.record(self._ev("A", "local", a1, 1.0))
        log.record(self._ev("A", "local", a2, 2.0))
        assert log.concurrent_pairs() == []

    def test_find_conflicts_on_key(self):
        log = EventLog()
        a = VectorClock("A", 2).increment()
        b = VectorClock("B", 2).increment()
        ea = self._ev("A", "local", a, 1.0, data={"x": 1})
        eb = self._ev("B", "local", b, 2.0, data={"x": 2})
        log.record(ea)
        log.record(eb)
        conflicts = log.find_conflicts("x")
        assert len(conflicts) == 1

    def test_find_conflicts_ignores_non_overlapping_keys(self):
        log = EventLog()
        a = VectorClock("A", 2).increment()
        b = VectorClock("B", 2).increment()
        log.record(self._ev("A", "local", a, 1.0, data={"x": 1}))
        log.record(self._ev("B", "local", b, 2.0, data={"y": 2}))
        assert log.find_conflicts("x") == []
        assert log.find_conflicts("y") == []

    def test_find_conflicts_handles_non_dict_data(self):
        log = EventLog()
        a = VectorClock("A", 2).increment()
        b = VectorClock("B", 2).increment()
        log.record(self._ev("A", "local", a, 1.0, data="raw"))
        log.record(self._ev("B", "local", b, 2.0, data=None))
        assert log.find_conflicts("x") == []

    def test_empty_log(self):
        log = EventLog()
        assert log.causal_order() == []
        assert log.concurrent_pairs() == []
        assert log.find_conflicts("k") == []


# --- Node ----------------------------------------------------------------

class TestNode:
    def test_local_event_increments_clock(self):
        log = EventLog()
        n = Node("A", 2, log)
        ev = n.local_event({"x": 1})
        assert ev.event_type == "local"
        assert ev.node_id == "A"
        assert ev.clock.counters["A"] == 1
        assert ev in log.events

    def test_send_increments_and_returns_message(self):
        log = EventLog()
        n = Node("A", 2, log)
        ev, msg = n.send({"payload": 42})
        assert ev.event_type == "send"
        assert ev.clock.counters["A"] == 1
        assert isinstance(msg, Message)
        assert msg.data == {"payload": 42}
        assert msg.clock == ev.clock

    def test_receive_merges_clock(self):
        log = EventLog()
        a = Node("A", 2, log)
        b = Node("B", 2, log)
        a.local_event()
        send_ev, msg = a.send({"hi": True})
        recv_ev = b.receive(msg)
        assert recv_ev.event_type == "receive"
        assert recv_ev.clock.counters["B"] == 1
        assert recv_ev.clock.counters["A"] == send_ev.clock.counters["A"]
        assert send_ev.clock.happens_before(recv_ev.clock)

    def test_receive_data_present_on_event(self):
        log = EventLog()
        a = Node("A", 2, log)
        b = Node("B", 2, log)
        _, msg = a.send({"k": "v"})
        ev = b.receive(msg)
        assert ev.data == {"k": "v"}


# --- Simulation ----------------------------------------------------------

class TestSimulation:
    def test_basic_local_events(self):
        sim = Simulation(["A", "B"])
        e1 = sim.local_event("A", {"v": 1})
        e2 = sim.local_event("B", {"v": 2})
        history = sim.get_history()
        assert e1 in history and e2 in history
        assert e1.clock.is_concurrent(e2.clock)

    def test_send_and_deliver(self):
        sim = Simulation(["A", "B"])
        send_ev = sim.send_message("A", "B", {"hello": 1})
        recv_ev = sim.deliver_message("B")
        assert send_ev.event_type == "send"
        assert recv_ev.event_type == "receive"
        assert send_ev.clock.happens_before(recv_ev.clock)
        assert recv_ev.data == {"hello": 1}

    def test_deliver_with_no_message_raises(self):
        sim = Simulation(["A", "B"])
        with pytest.raises((IndexError, KeyError, ValueError)):
            sim.deliver_message("B")

    def test_get_log_returns_event_log(self):
        sim = Simulation(["A"])
        sim.local_event("A")
        log = sim.get_log()
        assert isinstance(log, EventLog)
        assert len(log.events) == 1

    def test_no_messages_only_locals_all_concurrent(self):
        sim = Simulation(["A", "B", "C"])
        ea = sim.local_event("A")
        eb = sim.local_event("B")
        ec = sim.local_event("C")
        assert ea.clock.is_concurrent(eb.clock)
        assert ea.clock.is_concurrent(ec.clock)
        assert eb.clock.is_concurrent(ec.clock)
        pairs = sim.get_log().concurrent_pairs()
        assert len(pairs) == 3

    def test_single_node_sim(self):
        sim = Simulation(["only"])
        e1 = sim.local_event("only")
        e2 = sim.local_event("only")
        assert e1.clock.happens_before(e2.clock)
        assert sim.get_log().concurrent_pairs() == []

    def test_three_node_interleaved(self):
        sim = Simulation(["A", "B", "C"])
        a_local = sim.local_event("A", {"step": 1})
        sim.send_message("A", "B", {"from": "A"})
        b_recv = sim.deliver_message("B")
        b_local = sim.local_event("B", {"step": 2})
        sim.send_message("B", "C", {"from": "B"})
        c_recv = sim.deliver_message("C")
        assert a_local.clock.happens_before(b_recv.clock)
        assert b_recv.clock.happens_before(b_local.clock)
        assert b_local.clock.happens_before(c_recv.clock)
        assert a_local.clock.happens_before(c_recv.clock)

    def test_causal_order_after_complex_run(self):
        sim = Simulation(["A", "B"])
        sim.local_event("A")
        sim.send_message("A", "B", {"k": 1})
        sim.deliver_message("B")
        sim.local_event("B")
        ordered = sim.get_log().causal_order()
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                assert not ordered[j].clock.happens_before(ordered[i].clock)

    def test_concurrent_writes_conflict_detection(self):
        sim = Simulation(["A", "B"])
        sim.local_event("A", {"key": "x", "value": "a-wins"})
        sim.local_event("B", {"key": "x", "value": "b-wins"})
        conflicts = sim.get_log().find_conflicts("value")
        assert len(conflicts) == 1

    def test_no_conflict_when_writes_are_causally_ordered(self):
        sim = Simulation(["A", "B"])
        sim.local_event("A", {"value": 1})
        sim.send_message("A", "B", {"sync": True})
        sim.deliver_message("B")
        sim.local_event("B", {"value": 2})
        assert sim.get_log().find_conflicts("value") == []

    def test_get_history_preserves_record_order(self):
        sim = Simulation(["A", "B"])
        e1 = sim.local_event("A")
        e2 = sim.local_event("B")
        e3 = sim.local_event("A")
        assert sim.get_history() == [e1, e2, e3]
