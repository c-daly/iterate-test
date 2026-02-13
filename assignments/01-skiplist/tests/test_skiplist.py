"""Tests for the probabilistic skip list implementation."""

import random
import pytest

from skiplist import SkipList


class TestInsertAndSearch:
    def test_insert_and_search(self):
        """Insert 5 items, verify search returns correct values."""
        sl = SkipList()
        items = [(10, "a"), (20, "b"), (30, "c"), (40, "d"), (50, "e")]
        for k, v in items:
            sl.insert(k, v)
        for k, v in items:
            assert sl.search(k) == v

    def test_insert_update(self):
        """Same key twice: value updated, len unchanged."""
        sl = SkipList()
        sl.insert(5, "old")
        sl.insert(5, "new")
        assert sl.search(5) == "new"
        assert len(sl) == 1


class TestDelete:
    def test_delete_existing(self):
        """Insert then delete: True returned, search returns None."""
        sl = SkipList()
        sl.insert(1, "x")
        assert sl.delete(1) is True
        assert sl.search(1) is None

    def test_delete_nonexistent(self):
        """False on empty list and missing key."""
        sl = SkipList()
        assert sl.delete(99) is False
        sl.insert(1, "x")
        assert sl.delete(99) is False


class TestRangeQuery:
    def test_range_query(self):
        """10 items, verify sorted subset within range."""
        sl = SkipList()
        for i in range(10):
            sl.insert(i, str(i))
        result = sl.range_query(3, 7)
        assert result == [(3, "3"), (4, "4"), (5, "5"), (6, "6"), (7, "7")]

    def test_range_query_empty(self):
        """Range outside all keys returns empty list."""
        sl = SkipList()
        for i in range(10):
            sl.insert(i, str(i))
        assert sl.range_query(100, 200) == []


class TestLenAndContains:
    def test_len_and_contains(self):
        """After inserts and deletes, len and contains are correct."""
        sl = SkipList()
        assert len(sl) == 0
        sl.insert(1, "a")
        sl.insert(2, "b")
        sl.insert(3, "c")
        assert len(sl) == 3
        assert 1 in sl
        assert 2 in sl
        assert 4 not in sl
        sl.delete(2)
        assert len(sl) == 2
        assert 2 not in sl


class TestIterSortedOrder:
    def test_iter_sorted_order(self):
        """Random order insert, iteration yields sorted (key, value) pairs."""
        sl = SkipList()
        keys = list(range(20))
        random.shuffle(keys)
        for k in keys:
            sl.insert(k, k * 10)
        result = list(sl)
        expected = [(k, k * 10) for k in sorted(keys)]
        assert result == expected


class TestEdgeCases:
    def test_empty_operations(self):
        """All operations on an empty skip list."""
        sl = SkipList()
        assert len(sl) == 0
        assert sl.search(1) is None
        assert sl.delete(1) is False
        assert 1 not in sl
        assert list(sl) == []
        assert sl.range_query(0, 100) == []

    def test_single_element(self):
        """One item: all operations work correctly."""
        sl = SkipList()
        sl.insert(42, "only")
        assert len(sl) == 1
        assert sl.search(42) == "only"
        assert 42 in sl
        assert list(sl) == [(42, "only")]
        assert sl.range_query(0, 100) == [(42, "only")]
        assert sl.range_query(43, 100) == []
        assert sl.delete(42) is True
        assert len(sl) == 0
        assert sl.search(42) is None


class TestStress:
    def test_stress_1000(self):
        """1000 random keys, delete half, verify remaining."""
        sl = SkipList()
        random.seed(12345)
        keys = random.sample(range(10000), 1000)
        for k in keys:
            sl.insert(k, k)
        assert len(sl) == 1000

        to_delete = keys[:500]
        remaining = set(keys[500:])
        for k in to_delete:
            assert sl.delete(k) is True
        assert len(sl) == 500

        for k in to_delete:
            assert sl.search(k) is None
        for k in remaining:
            assert sl.search(k) == k

        # Verify sorted iteration
        result_keys = [kv[0] for kv in sl]
        assert result_keys == sorted(remaining)


class TestRepr:
    def test_repr(self):
        """repr returns a non-empty string."""
        sl = SkipList()
        r = repr(sl)
        assert isinstance(r, str)
        assert len(r) > 0
        sl.insert(1, "a")
        r = repr(sl)
        assert isinstance(r, str)
        assert len(r) > 0
