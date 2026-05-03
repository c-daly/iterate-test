"""Tests for SkipList implementation."""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from skiplist import SkipList  # noqa: E402


class TestBasicCRUD:
    def test_empty_list(self):
        sl = SkipList()
        assert len(sl) == 0
        assert sl.search(1) is None
        assert 1 not in sl
        assert list(sl) == []

    def test_single_insert(self):
        sl = SkipList()
        sl.insert(5, "five")
        assert len(sl) == 1
        assert sl.search(5) == "five"
        assert 5 in sl

    def test_multiple_insert_and_search(self):
        sl = SkipList()
        for k, v in [(3, "c"), (1, "a"), (4, "d"), (5, "e")]:
            sl.insert(k, v)
        assert len(sl) == 4
        assert sl.search(1) == "a"
        assert sl.search(3) == "c"
        assert sl.search(4) == "d"
        assert sl.search(5) == "e"
        assert sl.search(99) is None

    def test_delete_existing(self):
        sl = SkipList()
        sl.insert(1, "a")
        sl.insert(2, "b")
        assert sl.delete(1) is True
        assert len(sl) == 1
        assert sl.search(1) is None
        assert 1 not in sl
        assert sl.search(2) == "b"

    def test_delete_nonexistent(self):
        sl = SkipList()
        sl.insert(1, "a")
        assert sl.delete(99) is False
        assert len(sl) == 1
        assert sl.search(1) == "a"

    def test_delete_from_empty(self):
        sl = SkipList()
        assert sl.delete(1) is False
        assert len(sl) == 0

    def test_contains(self):
        sl = SkipList()
        sl.insert(10, "ten")
        assert 10 in sl
        assert 11 not in sl
        sl.delete(10)
        assert 10 not in sl


class TestDuplicateKey:
    def test_duplicate_insert_updates_value(self):
        sl = SkipList()
        sl.insert(1, "a")
        sl.insert(1, "b")
        assert sl.search(1) == "b"
        assert len(sl) == 1

    def test_duplicate_insert_keeps_count(self):
        sl = SkipList()
        for _ in range(10):
            sl.insert(42, "value")
        assert len(sl) == 1
        assert sl.search(42) == "value"

    def test_update_then_delete(self):
        sl = SkipList()
        sl.insert(1, "first")
        sl.insert(1, "second")
        assert sl.delete(1) is True
        assert len(sl) == 0
        assert sl.search(1) is None


class TestRangeQuery:
    def _make(self):
        sl = SkipList()
        for k in [1, 3, 5, 7, 9, 11, 13]:
            sl.insert(k, k * 10)
        return sl

    def test_range_full(self):
        sl = self._make()
        assert sl.range_query(0, 100) == [
            (1, 10), (3, 30), (5, 50), (7, 70), (9, 90), (11, 110), (13, 130)
        ]

    def test_range_inclusive_bounds(self):
        sl = self._make()
        assert sl.range_query(3, 9) == [(3, 30), (5, 50), (7, 70), (9, 90)]

    def test_range_inclusive_exact_match(self):
        sl = self._make()
        assert sl.range_query(5, 5) == [(5, 50)]

    def test_range_empty_no_keys_in_range(self):
        sl = self._make()
        assert sl.range_query(14, 100) == []
        assert sl.range_query(-10, 0) == []

    def test_range_empty_lo_greater_than_hi(self):
        sl = self._make()
        assert sl.range_query(10, 5) == []

    def test_range_on_empty_list(self):
        sl = SkipList()
        assert sl.range_query(0, 100) == []

    def test_range_partial_lo_below(self):
        sl = self._make()
        assert sl.range_query(-5, 5) == [(1, 10), (3, 30), (5, 50)]

    def test_range_partial_hi_above(self):
        sl = self._make()
        assert sl.range_query(11, 1000) == [(11, 110), (13, 130)]


class TestEdgeCases:
    def test_single_element_search_delete(self):
        sl = SkipList()
        sl.insert(7, "seven")
        assert sl.search(7) == "seven"
        assert sl.delete(7) is True
        assert len(sl) == 0
        assert sl.search(7) is None

    def test_delete_nonexistent_returns_false(self):
        sl = SkipList()
        sl.insert(1, "a")
        sl.insert(2, "b")
        sl.insert(3, "c")
        assert sl.delete(50) is False
        assert sl.delete(0) is False
        assert len(sl) == 3

    def test_negative_keys(self):
        sl = SkipList()
        sl.insert(-5, "neg")
        sl.insert(0, "zero")
        sl.insert(5, "pos")
        assert sl.search(-5) == "neg"
        assert list(sl) == [(-5, "neg"), (0, "zero"), (5, "pos")]

    def test_values_can_be_any_type(self):
        sl = SkipList()
        sl.insert(1, [1, 2, 3])
        sl.insert(2, {"a": 1})
        sl.insert(3, None)
        sl.insert(4, 3.14)
        assert sl.search(1) == [1, 2, 3]
        assert sl.search(2) == {"a": 1}
        assert 3 in sl
        assert sl.search(4) == 3.14

    def test_repr_returns_string(self):
        sl = SkipList()
        sl.insert(1, "a")
        r = repr(sl)
        assert isinstance(r, str)
        assert len(r) > 0


class TestIterator:
    def test_iter_sorted_order(self):
        sl = SkipList()
        keys = [5, 1, 9, 3, 7, 2, 8, 4, 6]
        for k in keys:
            sl.insert(k, str(k))
        assert list(sl) == [(i, str(i)) for i in range(1, 10)]

    def test_iter_after_deletes(self):
        sl = SkipList()
        for k in [1, 2, 3, 4, 5]:
            sl.insert(k, k)
        sl.delete(2)
        sl.delete(4)
        assert list(sl) == [(1, 1), (3, 3), (5, 5)]

    def test_iter_empty(self):
        sl = SkipList()
        assert list(sl) == []

    def test_iter_yields_tuples(self):
        sl = SkipList()
        sl.insert(1, "a")
        for item in sl:
            assert isinstance(item, tuple)
            assert len(item) == 2


class TestStress:
    def test_1000_insertions_correctness(self):
        random.seed(12345)
        sl = SkipList()
        keys = list(range(1000))
        random.shuffle(keys)
        for k in keys:
            sl.insert(k, k * 2)
        assert len(sl) == 1000
        for k in range(1000):
            assert sl.search(k) == k * 2
            assert k in sl

    def test_1500_iter_sorted(self):
        random.seed(99)
        sl = SkipList()
        keys = list(range(1500))
        random.shuffle(keys)
        for k in keys:
            sl.insert(k, k)
        observed = [k for k, _ in sl]
        assert observed == list(range(1500))

    def test_stress_mixed_ops(self):
        random.seed(7)
        sl = SkipList()
        ref = {}
        for _ in range(2000):
            op = random.choice(["ins", "ins", "del", "search"])
            k = random.randint(0, 500)
            if op == "ins":
                v = random.random()
                sl.insert(k, v)
                ref[k] = v
            elif op == "del":
                expected = k in ref
                got = sl.delete(k)
                assert got == expected
                ref.pop(k, None)
            else:
                assert sl.search(k) == ref.get(k)
        assert len(sl) == len(ref)
        assert list(sl) == sorted(ref.items())

    def test_height_logarithmic(self):
        random.seed(42)
        sl = SkipList(max_level=16, p=0.5)
        for k in range(1024):
            sl.insert(k, k)
        assert len(sl) == 1024
        assert sl.search(0) == 0
        assert sl.search(1023) == 1023
        assert sl.search(512) == 512

    def test_range_query_large(self):
        random.seed(3)
        sl = SkipList()
        for k in range(0, 1000):
            sl.insert(k, k)
        result = sl.range_query(100, 200)
        assert result == [(k, k) for k in range(100, 201)]
        assert len(result) == 101


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
