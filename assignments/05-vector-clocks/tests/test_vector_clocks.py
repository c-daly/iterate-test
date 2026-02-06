"""Comprehensive tests for vector clocks and causal ordering."""

import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vector_clock import VectorClock
from src.events import Event, EventLog
from src.node import Node, Message
from src.simulation import Simulation


# ──────────────────────────────────────────────
# VectorClock tests
# ──────────────────────────────────────────────

class TestVectorClockInit:
    def test_new_clock_has_zero_counters(self):
        vc = VectorClock("A", 3)
        assert vc.clock == {"A": 0}

    def test_node_id_stored(self):
        vc = VectorClock("B", 2)
        assert vc.node_id == "B"


class TestVectorClockIncrement:
    def test_increment_advances_own_counter(self):
        vc = VectorClock("A", 2)
        vc2 = vc.increment()
        assert vc2.clock["A"] == 1

    def test_increment_returns_new_clock(self):
        vc = VectorClock("A", 2)
        vc2 = vc.increment()
        # Original should be unchanged
        assert vc.clock["A"] == 0
        assert vc2.clock["A"] == 1

    def test_multiple_increments(self):
        vc = VectorClock("A", 2)
        vc = vc.increment().increment().increment()
        assert vc.clock["A"] == 3


class TestVectorClockMerge:
    def test_merge_takes_elementwise_max(self):
        vc_a = VectorClock("A", 2)
        vc_a = vc_a.increment().increment()  # A:2

        vc_b = VectorClock("B", 2)
        vc_b = vc_b.increment()  # B:1

        merged = vc_a.merge(vc_b)
        # After merge: max of each, then increment local
        assert merged.clock["A"] >= 2
        assert merged.clock.get("B", 0) >= 1

    def test_merge_increments_local_counter(self):
        vc_a = VectorClock("A", 2)
        vc_a = vc_a.increment()  # A:1

        vc_b = VectorClock("B", 2)
        vc_b = vc_b.increment()  # B:1

        merged = vc_a.merge(vc_b)
        # merge should increment A's counter after taking max
        assert merged.clock["A"] == 2  # was 1, +1 for merge

    def test_merge_returns_new_clock(self):
        vc_a = VectorClock("A", 2)
        vc_b = VectorClock("B", 2)
        merged = vc_a.merge(vc_b)
        assert merged is not vc_a
        assert merged is not vc_b


class TestVectorClockHappensBefore:
    def test_incremented_happens_after_original(self):
        vc = VectorClock("A", 2)
        vc2 = vc.increment()
        assert vc.happens_before(vc2)

    def test_not_happens_before_self(self):
        """happens_before is irreflexive."""
        vc = VectorClock("A", 2)
        assert not vc.happens_before(vc)

    def test_concurrent_events_neither_happens_before(self):
        vc_a = VectorClock("A", 2)
        vc_a = vc_a.increment()
        vc_b = VectorClock("B", 2)
        vc_b = vc_b.increment()
        assert not vc_a.happens_before(vc_b)
        assert not vc_b.happens_before(vc_a)

    def test_transitivity(self):
        vc1 = VectorClock("A", 2)
        vc2 = vc1.increment()
        vc3 = vc2.increment()
        assert vc1.happens_before(vc3)

    def test_happens_before_after_merge(self):
        vc_a = VectorClock("A", 2)
        vc_a = vc_a.increment()

        vc_b = VectorClock("B", 2)
        vc_b = vc_b.increment()

        merged = vc_b.merge(vc_a)
        # vc_a should happen before merged (merged incorporates vc_a)
        assert vc_a.happens_before(merged)


class TestVectorClockConcurrent:
    def test_independent_events_are_concurrent(self):
        vc_a = VectorClock("A", 2)
        vc_a = vc_a.increment()
        vc_b = VectorClock("B", 2)
        vc_b = vc_b.increment()
        assert vc_a.is_concurrent(vc_b)
        assert vc_b.is_concurrent(vc_a)

    def test_causally_related_not_concurrent(self):
        vc = VectorClock("A", 2)
        vc2 = vc.increment()
        assert not vc.is_concurrent(vc2)
        assert not vc2.is_concurrent(vc)

    def test_equal_clocks_not_concurrent(self):
        vc1 = VectorClock("A", 2)
        vc2 = VectorClock("A", 2)
        assert not vc1.is_concurrent(vc2)


class TestVectorClockComparison:
    def test_eq_same_clocks(self):
        vc1 = VectorClock("A", 2)
        vc2 = VectorClock("A", 2)
        assert vc1 == vc2

    def test_eq_different_clocks(self):
        vc1 = VectorClock("A", 2)
        vc2 = vc1.increment()
        assert vc1 != vc2

    def test_le_less_or_equal(self):
        vc1 = VectorClock("A", 2)
        vc2 = vc1.increment()
        assert vc1 <= vc2

    def test_le_equal(self):
        vc1 = VectorClock("A", 2)
        vc2 = VectorClock("A", 2)
        assert vc1 <= vc2

    def test_lt_strictly_less(self):
        vc1 = VectorClock("A", 2)
        vc2 = vc1.increment()
        assert vc1 < vc2

    def test_lt_not_equal(self):
        vc1 = VectorClock("A", 2)
        vc2 = VectorClock("A", 2)
        assert not (vc1 < vc2)

    def test_eq_with_non_vectorclock_returns_false(self):
        vc = VectorClock("A", 2)
        assert vc != "not a clock"


class TestVectorClockRepr:
    def test_repr_is_string(self):
        vc = VectorClock("A", 2)
        r = repr(vc)
        assert isinstance(r, str)
        assert "A" in r


# ──────────────────────────────────────────────
# EventLog tests
# ──────────────────────────────────────────────

class TestEventLog:
    def test_record_and_retrieve(self):
        log = EventLog()
        vc = VectorClock("A", 2)
        vc = vc.increment()
        event = Event(node_id="A", event_type="local", clock=vc, timestamp=1.0)
        log.record(event)
        ordered = log.causal_order()
        assert len(ordered) == 1
        assert ordered[0] is event

    def test_causal_order_respects_happens_before(self):
        log = EventLog()
        vc1 = VectorClock("A", 2)
        vc1 = vc1.increment()  # A:1
        vc2 = vc1.increment()  # A:2

        e1 = Event(node_id="A", event_type="local", clock=vc1, timestamp=1.0)
        e2 = Event(node_id="A", event_type="local", clock=vc2, timestamp=2.0)

        # Record in reverse order
        log.record(e2)
        log.record(e1)

        ordered = log.causal_order()
        assert ordered[0] is e1
        assert ordered[1] is e2

    def test_concurrent_pairs_detected(self):
        log = EventLog()
        vc_a = VectorClock("A", 2)
        vc_a = vc_a.increment()
        vc_b = VectorClock("B", 2)
        vc_b = vc_b.increment()

        e1 = Event(node_id="A", event_type="local", clock=vc_a, timestamp=1.0)
        e2 = Event(node_id="B", event_type="local", clock=vc_b, timestamp=2.0)

        log.record(e1)
        log.record(e2)

        pairs = log.concurrent_pairs()
        assert len(pairs) == 1
        pair = pairs[0]
        assert (e1 in pair) and (e2 in pair)

    def test_no_concurrent_pairs_for_causal_events(self):
        log = EventLog()
        vc1 = VectorClock("A", 2)
        vc1 = vc1.increment()
        vc2 = vc1.increment()

        e1 = Event(node_id="A", event_type="local", clock=vc1, timestamp=1.0)
        e2 = Event(node_id="A", event_type="local", clock=vc2, timestamp=2.0)

        log.record(e1)
        log.record(e2)

        pairs = log.concurrent_pairs()
        assert len(pairs) == 0

    def test_find_conflicts_on_same_key(self):
        log = EventLog()
        vc_a = VectorClock("A", 2)
        vc_a = vc_a.increment()
        vc_b = VectorClock("B", 2)
        vc_b = vc_b.increment()

        e1 = Event(
            node_id="A", event_type="local", clock=vc_a,
            timestamp=1.0, data={"key": "x", "value": 1},
        )
        e2 = Event(
            node_id="B", event_type="local", clock=vc_b,
            timestamp=2.0, data={"key": "x", "value": 2},
        )

        log.record(e1)
        log.record(e2)

        conflicts = log.find_conflicts("x")
        assert len(conflicts) == 1

    def test_find_conflicts_no_conflict_on_different_keys(self):
        log = EventLog()
        vc_a = VectorClock("A", 2)
        vc_a = vc_a.increment()
        vc_b = VectorClock("B", 2)
        vc_b = vc_b.increment()

        e1 = Event(
            node_id="A", event_type="local", clock=vc_a,
            timestamp=1.0, data={"key": "x", "value": 1},
        )
        e2 = Event(
            node_id="B", event_type="local", clock=vc_b,
            timestamp=2.0, data={"key": "y", "value": 2},
        )

        log.record(e1)
        log.record(e2)

        conflicts = log.find_conflicts("x")
        assert len(conflicts) == 0

    def test_find_conflicts_no_conflict_when_causal(self):
        log = EventLog()
        vc1 = VectorClock("A", 2)
        vc1 = vc1.increment()
        vc2 = vc1.increment()

        e1 = Event(
            node_id="A", event_type="local", clock=vc1,
            timestamp=1.0, data={"key": "x", "value": 1},
        )
        e2 = Event(
            node_id="A", event_type="local", clock=vc2,
            timestamp=2.0, data={"key": "x", "value": 2},
        )

        log.record(e1)
        log.record(e2)

        conflicts = log.find_conflicts("x")
        assert len(conflicts) == 0

    def test_empty_log(self):
        log = EventLog()
        assert log.causal_order() == []
        assert log.concurrent_pairs() == []
        assert log.find_conflicts("any") == []


# ──────────────────────────────────────────────
# Node tests
# ──────────────────────────────────────────────

class TestNode:
    def test_local_event_increments_clock(self):
        log = EventLog()
        node = Node("A", 2, log)
        event = node.local_event(data="hello")
        assert event.node_id == "A"
        assert event.event_type == "local"
        assert event.clock.clock["A"] == 1
        assert event.data == "hello"

    def test_local_event_recorded_in_log(self):
        log = EventLog()
        node = Node("A", 2, log)
        node.local_event()
        assert len(log.causal_order()) == 1

    def test_send_returns_event_and_message(self):
        log = EventLog()
        node = Node("A", 2, log)
        event, message = node.send(data="payload")
        assert event.event_type == "send"
        assert event.node_id == "A"
        assert isinstance(message, Message)
        assert message.sender_id == "A"
        assert message.data == "payload"

    def test_send_increments_clock(self):
        log = EventLog()
        node = Node("A", 2, log)
        event, message = node.send()
        assert event.clock.clock["A"] == 1

    def test_receive_merges_clock(self):
        log = EventLog()
        node_a = Node("A", 2, log)
        node_b = Node("B", 2, log)

        _, msg = node_a.send(data="hi")
        msg.receiver_id = "B"
        event = node_b.receive(msg)

        assert event.event_type == "receive"
        assert event.node_id == "B"
        # B should have merged A's clock: B:1, A:1
        assert event.clock.clock["B"] >= 1
        assert event.clock.clock.get("A", 0) >= 1

    def test_receive_recorded_in_log(self):
        log = EventLog()
        node_a = Node("A", 2, log)
        node_b = Node("B", 2, log)
        _, msg = node_a.send()
        msg.receiver_id = "B"
        node_b.receive(msg)
        # Should have send event + receive event = 2
        assert len(log.causal_order()) == 2

    def test_multiple_local_events(self):
        log = EventLog()
        node = Node("A", 2, log)
        node.local_event()
        node.local_event()
        event = node.local_event()
        assert event.clock.clock["A"] == 3

    def test_send_event_has_timestamp(self):
        log = EventLog()
        node = Node("A", 2, log)
        event, _ = node.send()
        assert isinstance(event.timestamp, float)
        assert event.timestamp > 0


# ──────────────────────────────────────────────
# Simulation tests
# ──────────────────────────────────────────────

class TestSimulation:
    def test_create_simulation(self):
        sim = Simulation(["A", "B", "C"])
        log = sim.get_log()
        assert isinstance(log, EventLog)

    def test_local_event(self):
        sim = Simulation(["A", "B"])
        event = sim.local_event("A", data="work")
        assert event.node_id == "A"
        assert event.event_type == "local"
        assert event.data == "work"

    def test_send_and_deliver(self):
        sim = Simulation(["A", "B"])
        send_event = sim.send_message("A", "B", data="hello")
        assert send_event.event_type == "send"

        recv_event = sim.deliver_message("B")
        assert recv_event.event_type == "receive"
        assert recv_event.node_id == "B"

    def test_deliver_reflects_sender_clock(self):
        sim = Simulation(["A", "B"])
        sim.local_event("A")  # A:1
        sim.send_message("A", "B")  # A:2
        recv_event = sim.deliver_message("B")
        # B should know about A's clock
        assert recv_event.clock.clock.get("A", 0) >= 2

    def test_get_history_returns_all_events(self):
        sim = Simulation(["A", "B"])
        sim.local_event("A")
        sim.local_event("B")
        sim.send_message("A", "B")
        sim.deliver_message("B")
        history = sim.get_history()
        assert len(history) == 4

    def test_three_node_scenario(self):
        """Complex: 3 nodes with interleaved communication."""
        sim = Simulation(["A", "B", "C"])

        # A does local work
        sim.local_event("A")

        # A sends to B
        sim.send_message("A", "B")
        sim.deliver_message("B")

        # B does local work
        sim.local_event("B")

        # B sends to C
        sim.send_message("B", "C")
        sim.deliver_message("C")

        # C does local work
        sim.local_event("C")

        # A does more local work (concurrent with B->C and C)
        sim.local_event("A")

        history = sim.get_history()
        assert len(history) == 8

        log = sim.get_log()
        ordered = log.causal_order()
        assert len(ordered) == 8

        # The last A event should be concurrent with C's events
        concurrent = log.concurrent_pairs()
        assert len(concurrent) > 0

    def test_single_node_no_messages(self):
        """Edge case: only one node, no communication."""
        sim = Simulation(["solo"])
        sim.local_event("solo")
        sim.local_event("solo")
        sim.local_event("solo")

        history = sim.get_history()
        assert len(history) == 3

        log = sim.get_log()
        assert log.concurrent_pairs() == []

        ordered = log.causal_order()
        # All should be in order
        for i in range(len(ordered) - 1):
            assert ordered[i].clock.happens_before(ordered[i + 1].clock)

    def test_all_concurrent_events(self):
        """Edge case: multiple nodes, no communication -> all concurrent."""
        sim = Simulation(["A", "B", "C"])
        sim.local_event("A")
        sim.local_event("B")
        sim.local_event("C")

        log = sim.get_log()
        pairs = log.concurrent_pairs()
        # 3 choose 2 = 3 concurrent pairs
        assert len(pairs) == 3

    def test_conflict_detection_in_simulation(self):
        """Concurrent writes to same key should be detected as conflicts."""
        sim = Simulation(["A", "B"])
        sim.local_event("A", data={"key": "x", "value": 1})
        sim.local_event("B", data={"key": "x", "value": 2})

        log = sim.get_log()
        conflicts = log.find_conflicts("x")
        assert len(conflicts) == 1

    def test_no_conflict_after_sync(self):
        """After synchronizing, writes to same key should not conflict."""
        sim = Simulation(["A", "B"])
        sim.local_event("A", data={"key": "x", "value": 1})
        sim.send_message("A", "B")
        sim.deliver_message("B")
        # B writes after receiving from A -> causally after
        sim.local_event("B", data={"key": "x", "value": 2})

        log = sim.get_log()
        conflicts = log.find_conflicts("x")
        assert len(conflicts) == 0

    def test_deliver_without_send_raises(self):
        """Delivering when no messages queued should raise."""
        sim = Simulation(["A", "B"])
        try:
            sim.deliver_message("B")
            assert False, "Expected an exception"
        except (IndexError, ValueError, KeyError):
            pass  # Any of these is acceptable
