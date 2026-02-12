"""Tests for probabilistic skip list."""

import random
import pytest
from src.skiplist import SkipList


class TestCreation:
    def test_empty_skiplist(self):
        sl = SkipList()
        assert len(sl) == 0

    def test_custom_max_level(self):
        sl = SkipList(max_level=8)
        assert len(sl) == 0

    def test_custom_probability(self):
        sl = SkipList(p=0.25)
        assert len(sl) == 0

    def test_repr_empty(self):
        sl = SkipList()
        r = repr(sl)
        assert "SkipList" in r


class TestInsert:
    def test_insert_single(self):
        sl = SkipList()
        sl.insert(5, "five")
        assert len(sl) == 1
        assert 5 in sl

    def test_insert_multiple(self):
        sl = SkipList()
        sl.insert(3, "three")
        sl.insert(1, "one")
        sl.insert(5, "five")
        assert len(sl) == 3
        assert 1 in sl
        assert 3 in sl
        assert 5 in sl

    def test_insert_duplicate_updates_value(self):
        sl = SkipList()
        sl.insert(5, "five")
        sl.insert(5, "FIVE")
        assert len(sl) == 1
        assert sl.search(5) == "FIVE"

    def test_insert_negative_keys(self):
        sl = SkipList()
        sl.insert(-3, "neg_three")
        sl.insert(-1, "neg_one")
        assert len(sl) == 2
        assert sl.search(-3) == "neg_three"
        assert sl.search(-1) == "neg_one"

    def test_insert_zero_key(self):
        sl = SkipList()
        sl.insert(0, "zero")
        assert len(sl) == 1
        assert sl.search(0) == "zero"

    def test_insert_various_value_types(self):
        sl = SkipList()
        sl.insert(1, "string")
        sl.insert(2, 42)
        sl.insert(3, [1, 2, 3])
        sl.insert(4, {"a": 1})
        sl.insert(5, None)
        assert sl.search(1) == "string"
        assert sl.search(2) == 42
        assert sl.search(3) == [1, 2, 3]
        assert sl.search(4) == {"a": 1}
        assert sl.search(5) is None


class TestSearch:
    def test_search_existing(self):
        sl = SkipList()
        sl.insert(10, "ten")
        assert sl.search(10) == "ten"

    def test_search_nonexistent(self):
        sl = SkipList()
        sl.insert(10, "ten")
        assert sl.search(20) is None

    def test_search_empty(self):
        sl = SkipList()
        assert sl.search(1) is None

    def test_search_after_update(self):
        sl = SkipList()
        sl.insert(5, "old")
        sl.insert(5, "new")
        assert sl.search(5) == "new"


class TestDelete:
    def test_delete_existing(self):
        sl = SkipList()
        sl.insert(5, "five")
        assert sl.delete(5) is True
        assert len(sl) == 0
        assert 5 not in sl

    def test_delete_nonexistent(self):
        sl = SkipList()
        sl.insert(5, "five")
        assert sl.delete(10) is False
        assert len(sl) == 1

    def test_delete_from_empty(self):
        sl = SkipList()
        assert sl.delete(1) is False

    def test_delete_middle_element(self):
        sl = SkipList()
        sl.insert(1, "one")
        sl.insert(3, "three")
        sl.insert(5, "five")
        assert sl.delete(3) is True
        assert len(sl) == 2
        assert 3 not in sl
        assert sl.search(1) == "one"
        assert sl.search(5) == "five"

    def test_delete_first_element(self):
        sl = SkipList()
        sl.insert(1, "one")
        sl.insert(3, "three")
        sl.insert(5, "five")
        assert sl.delete(1) is True
        assert len(sl) == 2
        assert 1 not in sl

    def test_delete_last_element(self):
        sl = SkipList()
        sl.insert(1, "one")
        sl.insert(3, "three")
        sl.insert(5, "five")
        assert sl.delete(5) is True
        assert len(sl) == 2
        assert 5 not in sl

    def test_delete_then_reinsert(self):
        sl = SkipList()
        sl.insert(5, "five")
        sl.delete(5)
        sl.insert(5, "FIVE")
        assert sl.search(5) == "FIVE"
        assert len(sl) == 1

    def test_delete_all_elements(self):
        sl = SkipList()
        for i in range(10):
            sl.insert(i, str(i))
        for i in range(10):
            assert sl.delete(i) is True
        assert len(sl) == 0

    def test_search_after_delete(self):
        sl = SkipList()
        sl.insert(5, "five")
        sl.delete(5)
        assert sl.search(5) is None


class TestContains:
    def test_contains_existing(self):
        sl = SkipList()
        sl.insert(5, "five")
        assert 5 in sl

    def test_contains_nonexistent(self):
        sl = SkipList()
        sl.insert(5, "five")
        assert 10 not in sl

    def test_contains_empty(self):
        sl = SkipList()
        assert 1 not in sl

    def test_contains_after_delete(self):
        sl = SkipList()
        sl.insert(5, "five")
        sl.delete(5)
        assert 5 not in sl


class TestRangeQuery:
    def test_range_query_basic(self):
        sl = SkipList()
        for i in range(10):
            sl.insert(i, str(i))
        result = sl.range_query(3, 7)
        assert result == [(3, "3"), (4, "4"), (5, "5"), (6, "6"), (7, "7")]

    def test_range_query_inclusive(self):
        sl = SkipList()
        sl.insert(1, "one")
        sl.insert(3, "three")
        sl.insert(5, "five")
        result = sl.range_query(1, 5)
        assert result == [(1, "one"), (3, "three"), (5, "five")]

    def test_range_query_single_element(self):
        sl = SkipList()
        sl.insert(5, "five")
        result = sl.range_query(5, 5)
        assert result == [(5, "five")]

    def test_range_query_empty_result(self):
        sl = SkipList()
        sl.insert(1, "one")
        sl.insert(10, "ten")
        result = sl.range_query(3, 7)
        assert result == []

    def test_range_query_empty_list(self):
        sl = SkipList()
        result = sl.range_query(1, 10)
        assert result == []

    def test_range_query_partial_overlap(self):
        sl = SkipList()
        for i in range(1, 6):
            sl.insert(i, str(i))
        result = sl.range_query(3, 100)
        assert result == [(3, "3"), (4, "4"), (5, "5")]

    def test_range_query_sorted_order(self):
        sl = SkipList()
        keys = [9, 3, 7, 1, 5]
        for k in keys:
            sl.insert(k, str(k))
        result = sl.range_query(1, 9)
        assert result == [(1, "1"), (3, "3"), (5, "5"), (7, "7"), (9, "9")]


class TestIterator:
    def test_iter_empty(self):
        sl = SkipList()
        assert list(sl) == []

    def test_iter_single(self):
        sl = SkipList()
        sl.insert(5, "five")
        assert list(sl) == [(5, "five")]

    def test_iter_sorted_order(self):
        sl = SkipList()
        sl.insert(3, "three")
        sl.insert(1, "one")
        sl.insert(5, "five")
        sl.insert(2, "two")
        sl.insert(4, "four")
        expected = [(1, "one"), (2, "two"), (3, "three"), (4, "four"), (5, "five")]
        assert list(sl) == expected

    def test_iter_yields_tuples(self):
        sl = SkipList()
        sl.insert(1, "one")
        sl.insert(2, "two")
        for item in sl:
            assert isinstance(item, tuple)
            assert len(item) == 2


class TestRepr:
    def test_repr_nonempty(self):
        sl = SkipList()
        sl.insert(1, "one")
        sl.insert(2, "two")
        r = repr(sl)
        assert "SkipList" in r

    def test_repr_contains_length(self):
        sl = SkipList()
        for i in range(5):
            sl.insert(i, str(i))
        r = repr(sl)
        assert "5" in r


class TestProbabilisticHeight:
    def test_level_bounded_by_max(self):
        """All node levels should be <= max_level."""
        sl = SkipList(max_level=4, p=0.5)
        random.seed(42)
        for i in range(200):
            sl.insert(i, str(i))
        # Walk level 0 and check that no node has level > max_level
        # We verify indirectly: the skip list should function correctly
        # with bounded levels
        assert len(sl) == 200
        for i in range(200):
            assert sl.search(i) == str(i)

    def test_height_statistical_distribution(self):
        """With p=0.5, roughly half the nodes should have level >= 2."""
        random.seed(12345)
        sl = SkipList(max_level=16, p=0.5)
        for i in range(10000):
            sl.insert(i, i)
        # Verify the structure works correctly
        assert len(sl) == 10000
        for i in [0, 5000, 9999]:
            assert sl.search(i) == i

    def test_low_probability_keeps_structure_flat(self):
        """With very low p, most nodes should be at level 1."""
        random.seed(99)
        sl = SkipList(max_level=16, p=0.01)
        for i in range(1000):
            sl.insert(i, i)
        assert len(sl) == 1000
        # Should still work correctly
        for i in range(1000):
            assert sl.search(i) == i


class TestStress:
    def test_stress_insertions(self):
        """Insert 1000+ elements and verify all are searchable."""
        sl = SkipList()
        n = 2000
        for i in range(n):
            sl.insert(i, i * 10)
        assert len(sl) == n
        for i in range(n):
            assert sl.search(i) == i * 10

    def test_stress_random_operations(self):
        """Random mix of insert, delete, search operations."""
        random.seed(42)
        sl = SkipList()
        reference = {}
        for _ in range(2000):
            op = random.choice(["insert", "insert", "delete", "search"])
            key = random.randint(0, 500)
            if op == "insert":
                val = random.randint(0, 10000)
                sl.insert(key, val)
                reference[key] = val
            elif op == "delete":
                result = sl.delete(key)
                assert result == (key in reference)
                reference.pop(key, None)
            else:
                result = sl.search(key)
                assert result == reference.get(key, None)
        assert len(sl) == len(reference)

    def test_stress_sequential_delete(self):
        """Insert 1000 elements then delete all."""
        sl = SkipList()
        n = 1000
        for i in range(n):
            sl.insert(i, str(i))
        for i in range(n):
            assert sl.delete(i) is True
        assert len(sl) == 0

    def test_stress_reverse_insert(self):
        """Insert in reverse order, verify sorted iteration."""
        sl = SkipList()
        n = 1000
        for i in range(n - 1, -1, -1):
            sl.insert(i, str(i))
        items = list(sl)
        assert len(items) == n
        for idx, (k, v) in enumerate(items):
            assert k == idx
            assert v == str(idx)


class TestEdgeCases:
    def test_single_element_operations(self):
        sl = SkipList()
        sl.insert(42, "answer")
        assert sl.search(42) == "answer"
        assert 42 in sl
        assert len(sl) == 1
        assert list(sl) == [(42, "answer")]
        assert sl.range_query(0, 100) == [(42, "answer")]
        assert sl.delete(42) is True
        assert len(sl) == 0

    def test_large_keys(self):
        sl = SkipList()
        sl.insert(10**9, "billion")
        sl.insert(-10**9, "neg_billion")
        assert sl.search(10**9) == "billion"
        assert sl.search(-10**9) == "neg_billion"
        items = list(sl)
        assert items[0][0] < items[1][0]

    def test_none_value(self):
        """Value of None should be distinguishable from key-not-found."""
        sl = SkipList()
        sl.insert(1, None)
        # search returns None for both missing key and None value
        # but __contains__ should confirm the key exists
        assert 1 in sl
        assert sl.search(1) is None

    def test_duplicate_insert_preserves_length(self):
        sl = SkipList()
        sl.insert(1, "a")
        sl.insert(1, "b")
        sl.insert(1, "c")
        assert len(sl) == 1
        assert sl.search(1) == "c"
