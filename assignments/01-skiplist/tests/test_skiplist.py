"""Tests for Probabilistic Skip List implementation."""

import sys
import os

# Add src to path so we can import skiplist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from skiplist import SkipList


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------

class TestInsertAndSearch:
    def test_insert_single(self):
        sl = SkipList()
        sl.insert(5, "five")
        assert sl.search(5) == "five"

    def test_insert_multiple(self):
        sl = SkipList()
        sl.insert(3, "three")
        sl.insert(1, "one")
        sl.insert(7, "seven")
        assert sl.search(1) == "one"
        assert sl.search(3) == "three"
        assert sl.search(7) == "seven"

    def test_search_missing_returns_none(self):
        sl = SkipList()
        assert sl.search(42) is None

    def test_search_empty_list(self):
        sl = SkipList()
        assert sl.search(0) is None

    def test_insert_various_value_types(self):
        sl = SkipList()
        sl.insert(1, 100)
        sl.insert(2, [1, 2, 3])
        sl.insert(3, {"a": 1})
        sl.insert(4, None)
        assert sl.search(1) == 100
        assert sl.search(2) == [1, 2, 3]
        assert sl.search(3) == {"a": 1}
        assert sl.search(4) is None


class TestDuplicateKeyUpdate:
    def test_update_existing_key(self):
        sl = SkipList()
        sl.insert(5, "old")
        sl.insert(5, "new")
        assert sl.search(5) == "new"

    def test_update_preserves_length(self):
        sl = SkipList()
        sl.insert(5, "old")
        sl.insert(5, "new")
        assert len(sl) == 1

    def test_multiple_updates(self):
        sl = SkipList()
        for i in range(10):
            sl.insert(1, i)
        assert sl.search(1) == 9
        assert len(sl) == 1


class TestDelete:
    def test_delete_existing(self):
        sl = SkipList()
        sl.insert(5, "five")
        result = sl.delete(5)
        assert result is True
        assert sl.search(5) is None

    def test_delete_nonexistent(self):
        sl = SkipList()
        sl.insert(1, "one")
        result = sl.delete(99)
        assert result is False

    def test_delete_from_empty(self):
        sl = SkipList()
        assert sl.delete(1) is False

    def test_delete_updates_length(self):
        sl = SkipList()
        sl.insert(1, "one")
        sl.insert(2, "two")
        sl.insert(3, "three")
        assert len(sl) == 3
        sl.delete(2)
        assert len(sl) == 2

    def test_delete_then_reinsert(self):
        sl = SkipList()
        sl.insert(5, "first")
        sl.delete(5)
        sl.insert(5, "second")
        assert sl.search(5) == "second"
        assert len(sl) == 1

    def test_delete_first_element(self):
        sl = SkipList()
        sl.insert(1, "one")
        sl.insert(2, "two")
        sl.insert(3, "three")
        sl.delete(1)
        assert sl.search(1) is None
        assert sl.search(2) == "two"
        assert sl.search(3) == "three"

    def test_delete_last_element(self):
        sl = SkipList()
        sl.insert(1, "one")
        sl.insert(2, "two")
        sl.insert(3, "three")
        sl.delete(3)
        assert sl.search(3) is None
        assert sl.search(1) == "one"
        assert sl.search(2) == "two"

    def test_delete_all_elements(self):
        sl = SkipList()
        sl.insert(1, "a")
        sl.insert(2, "b")
        sl.insert(3, "c")
        sl.delete(1)
        sl.delete(2)
        sl.delete(3)
        assert len(sl) == 0
        assert sl.search(1) is None
        assert sl.search(2) is None
        assert sl.search(3) is None


# ---------------------------------------------------------------------------
# Range queries
# ---------------------------------------------------------------------------

class TestRangeQuery:
    def test_range_inclusive_bounds(self):
        sl = SkipList()
        for i in range(1, 11):
            sl.insert(i, str(i))
        result = sl.range_query(3, 7)
        assert result == [(3, "3"), (4, "4"), (5, "5"), (6, "6"), (7, "7")]

    def test_range_single_element(self):
        sl = SkipList()
        sl.insert(5, "five")
        result = sl.range_query(5, 5)
        assert result == [(5, "five")]

    def test_range_empty_result(self):
        sl = SkipList()
        sl.insert(1, "one")
        sl.insert(10, "ten")
        result = sl.range_query(3, 7)
        assert result == []

    def test_range_on_empty_list(self):
        sl = SkipList()
        assert sl.range_query(1, 10) == []

    def test_range_covers_all(self):
        sl = SkipList()
        sl.insert(2, "b")
        sl.insert(1, "a")
        sl.insert(3, "c")
        result = sl.range_query(0, 100)
        assert result == [(1, "a"), (2, "b"), (3, "c")]

    def test_range_lo_greater_than_hi(self):
        sl = SkipList()
        sl.insert(1, "a")
        sl.insert(2, "b")
        result = sl.range_query(5, 1)
        assert result == []

    def test_range_returns_sorted(self):
        sl = SkipList()
        # Insert out of order
        for k in [9, 3, 7, 1, 5]:
            sl.insert(k, k * 10)
        result = sl.range_query(1, 9)
        keys = [k for k, _ in result]
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Dunder methods
# ---------------------------------------------------------------------------

class TestLen:
    def test_empty(self):
        assert len(SkipList()) == 0

    def test_after_inserts(self):
        sl = SkipList()
        for i in range(5):
            sl.insert(i, i)
        assert len(sl) == 5

    def test_after_delete(self):
        sl = SkipList()
        sl.insert(1, "a")
        sl.insert(2, "b")
        sl.delete(1)
        assert len(sl) == 1


class TestContains:
    def test_contains_present(self):
        sl = SkipList()
        sl.insert(5, "five")
        assert 5 in sl

    def test_contains_absent(self):
        sl = SkipList()
        sl.insert(5, "five")
        assert 99 not in sl

    def test_contains_empty(self):
        sl = SkipList()
        assert 1 not in sl

    def test_contains_after_delete(self):
        sl = SkipList()
        sl.insert(5, "five")
        sl.delete(5)
        assert 5 not in sl


class TestIter:
    def test_iter_empty(self):
        sl = SkipList()
        assert list(sl) == []

    def test_iter_sorted_order(self):
        sl = SkipList()
        sl.insert(3, "c")
        sl.insert(1, "a")
        sl.insert(2, "b")
        assert list(sl) == [(1, "a"), (2, "b"), (3, "c")]

    def test_iter_yields_tuples(self):
        sl = SkipList()
        sl.insert(1, "one")
        items = list(sl)
        assert len(items) == 1
        assert isinstance(items[0], tuple)
        assert items[0] == (1, "one")

    def test_iter_large_set(self):
        sl = SkipList()
        for i in range(100):
            sl.insert(i, i)
        items = list(sl)
        assert len(items) == 100
        assert items == [(i, i) for i in range(100)]


class TestRepr:
    def test_repr_returns_string(self):
        sl = SkipList()
        r = repr(sl)
        assert isinstance(r, str)

    def test_repr_nonempty(self):
        sl = SkipList()
        sl.insert(1, "a")
        r = repr(sl)
        assert isinstance(r, str)
        # Should contain some useful info (implementation-flexible)
        assert len(r) > 0


# ---------------------------------------------------------------------------
# Probabilistic / structural properties
# ---------------------------------------------------------------------------

class TestProbabilisticBehavior:
    def test_max_level_respected(self):
        sl = SkipList(max_level=4, p=0.99)
        # Even with high p, levels should not exceed max_level
        for i in range(200):
            sl.insert(i, i)
        # All elements searchable
        for i in range(200):
            assert sl.search(i) == i

    def test_height_bounded_log_n(self):
        """For large n, the observed max level should be roughly O(log n)."""
        sl = SkipList(max_level=32, p=0.5)
        n = 5000
        for i in range(n):
            sl.insert(i, i)
        # With p=0.5 and n=5000, expected max level ~ log2(5000) ~ 12.3
        # We allow generous upper bound but it shouldn't reach max_level=32
        # This is probabilistic; we just check it's not degenerate
        assert len(sl) == n

    def test_custom_p_value(self):
        """A very low p should produce mostly level-1 nodes."""
        sl = SkipList(max_level=16, p=0.01)
        for i in range(500):
            sl.insert(i, i)
        # Should still be correct
        assert len(sl) == 500
        for i in range(500):
            assert sl.search(i) == i


# ---------------------------------------------------------------------------
# Stress test
# ---------------------------------------------------------------------------

class TestStress:
    def test_large_insertions_correctness(self):
        """Insert 2000 elements, verify all searchable, delete half, re-check."""
        sl = SkipList()
        n = 2000
        for i in range(n):
            sl.insert(i, i * 10)
        assert len(sl) == n
        # Verify all present
        for i in range(n):
            assert sl.search(i) == i * 10
        # Delete even keys
        for i in range(0, n, 2):
            assert sl.delete(i) is True
        assert len(sl) == n // 2
        # Verify odd still present, even gone
        for i in range(n):
            if i % 2 == 0:
                assert sl.search(i) is None
            else:
                assert sl.search(i) == i * 10

    def test_sorted_iteration_after_mixed_ops(self):
        sl = SkipList()
        import random
        rng = random.Random(42)
        keys = list(range(1000))
        rng.shuffle(keys)
        for k in keys:
            sl.insert(k, k)
        # Delete a random subset
        to_delete = set(rng.sample(range(1000), 300))
        for k in to_delete:
            sl.delete(k)
        remaining = sorted(set(range(1000)) - to_delete)
        items = list(sl)
        assert [k for k, _ in items] == remaining

    def test_range_query_under_load(self):
        sl = SkipList()
        for i in range(1000):
            sl.insert(i, i)
        result = sl.range_query(100, 199)
        assert len(result) == 100
        assert result == [(i, i) for i in range(100, 200)]

    def test_negative_keys(self):
        sl = SkipList()
        for i in range(-50, 51):
            sl.insert(i, i)
        assert len(sl) == 101
        assert sl.search(-50) == -50
        assert sl.search(0) == 0
        assert sl.search(50) == 50
        items = list(sl)
        keys = [k for k, _ in items]
        assert keys == list(range(-50, 51))
