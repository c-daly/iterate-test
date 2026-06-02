"""Tests for the probabilistic SkipList (Assignment 1).

Covers the spec's stated expectations:
- Basic CRUD operations
- Duplicate key handling (update semantics)
- Range queries (inclusive bounds, empty ranges)
- Edge cases: empty list operations, single element, delete nonexistent
- Probabilistic behavior: large insertions maintain O(log n) height
- Iterator correctness and ordering
- Stress test: 1000+ insertions with correctness verification
"""

import math
import random

import pytest

from skiplist import SkipList


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def reset_random_seed():
    # Tests that call random.seed() mutate the global random state; save and
    # restore it so seeding cannot pollute other (order-dependent) tests.
    state = random.getstate()
    yield
    random.setstate(state)


@pytest.fixture
def sl():
    return SkipList()


def _populate(sl, pairs):
    for k, v in pairs:
        sl.insert(k, v)
    return sl


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_default_construction(self):
        sl = SkipList()
        assert len(sl) == 0

    def test_custom_params(self):
        sl = SkipList(max_level=8, p=0.25)
        assert sl.max_level == 8
        assert sl.p == 0.25
        assert len(sl) == 0

    def test_empty_is_falsey_via_len(self):
        sl = SkipList()
        assert not len(sl)


# --------------------------------------------------------------------------- #
# Basic CRUD
# --------------------------------------------------------------------------- #
class TestCRUD:
    def test_insert_then_search(self, sl):
        sl.insert(5, "five")
        assert sl.search(5) == "five"

    def test_insert_increases_len(self, sl):
        sl.insert(1, "a")
        sl.insert(2, "b")
        sl.insert(3, "c")
        assert len(sl) == 3

    def test_search_missing_returns_none(self, sl):
        sl.insert(1, "a")
        assert sl.search(999) is None

    def test_delete_existing_returns_true(self, sl):
        sl.insert(7, "seven")
        assert sl.delete(7) is True
        assert sl.search(7) is None
        assert len(sl) == 0

    def test_delete_nonexistent_returns_false(self, sl):
        sl.insert(1, "a")
        assert sl.delete(42) is False
        assert len(sl) == 1

    def test_delete_does_not_affect_other_keys(self, sl):
        _populate(sl, [(1, "a"), (2, "b"), (3, "c")])
        assert sl.delete(2) is True
        assert sl.search(1) == "a"
        assert sl.search(3) == "c"
        assert sl.search(2) is None
        assert len(sl) == 2

    def test_insert_after_delete(self, sl):
        sl.insert(5, "v1")
        assert sl.delete(5) is True
        sl.insert(5, "v2")
        assert sl.search(5) == "v2"
        assert len(sl) == 1

    def test_value_can_be_any_type(self, sl):
        obj = {"nested": [1, 2, 3]}
        sl.insert(1, obj)
        sl.insert(2, None)
        sl.insert(3, 3.14)
        assert sl.search(1) is obj
        assert sl.search(2) is None  # value is None, key exists
        assert 2 in sl  # distinguish missing key from None value
        assert sl.search(3) == 3.14


# --------------------------------------------------------------------------- #
# Duplicate key / update semantics
# --------------------------------------------------------------------------- #
class TestUpdateSemantics:
    def test_reinsert_updates_value(self, sl):
        sl.insert(10, "old")
        sl.insert(10, "new")
        assert sl.search(10) == "new"

    def test_reinsert_does_not_grow_len(self, sl):
        sl.insert(10, "old")
        sl.insert(10, "new")
        assert len(sl) == 1

    def test_repeated_updates(self, sl):
        for i in range(20):
            sl.insert(5, i)
        assert sl.search(5) == 19
        assert len(sl) == 1

    def test_update_preserves_other_entries(self, sl):
        _populate(sl, [(1, "a"), (2, "b"), (3, "c")])
        sl.insert(2, "B")
        assert list(sl) == [(1, "a"), (2, "B"), (3, "c")]
        assert len(sl) == 3


# --------------------------------------------------------------------------- #
# __contains__
# --------------------------------------------------------------------------- #
class TestContains:
    def test_contains_true(self, sl):
        sl.insert(3, "x")
        assert 3 in sl

    def test_contains_false(self, sl):
        sl.insert(3, "x")
        assert 4 not in sl

    def test_contains_after_delete(self, sl):
        sl.insert(3, "x")
        sl.delete(3)
        assert 3 not in sl

    def test_contains_empty(self, sl):
        assert 0 not in sl


# --------------------------------------------------------------------------- #
# Iterator / ordering
# --------------------------------------------------------------------------- #
class TestIteration:
    def test_iter_sorted_order(self, sl):
        _populate(sl, [(3, "c"), (1, "a"), (2, "b")])
        assert list(sl) == [(1, "a"), (2, "b"), (3, "c")]

    def test_iter_empty(self, sl):
        assert list(sl) == []

    def test_iter_yields_key_value_tuples(self, sl):
        sl.insert(1, "a")
        items = list(sl)
        assert items == [(1, "a")]
        assert isinstance(items[0], tuple)

    def test_iter_reflects_updates_and_deletes(self, sl):
        _populate(sl, [(1, "a"), (2, "b"), (3, "c"), (4, "d")])
        sl.insert(2, "B")
        sl.delete(3)
        assert list(sl) == [(1, "a"), (2, "B"), (4, "d")]

    def test_iter_with_negative_keys(self, sl):
        _populate(sl, [(0, "z"), (-5, "a"), (5, "b"), (-1, "c")])
        assert [k for k, _ in sl] == [-5, -1, 0, 5]

    def test_multiple_independent_iterations(self, sl):
        _populate(sl, [(2, "b"), (1, "a"), (3, "c")])
        first = list(sl)
        second = list(sl)
        assert first == second == [(1, "a"), (2, "b"), (3, "c")]


# --------------------------------------------------------------------------- #
# Range queries
# --------------------------------------------------------------------------- #
class TestRangeQuery:
    def test_range_inclusive_bounds(self, sl):
        _populate(sl, [(i, str(i)) for i in range(10)])
        result = sl.range_query(3, 6)
        assert result == [(3, "3"), (4, "4"), (5, "5"), (6, "6")]

    def test_range_lower_bound_inclusive(self, sl):
        _populate(sl, [(1, "a"), (2, "b"), (3, "c")])
        assert sl.range_query(1, 2) == [(1, "a"), (2, "b")]

    def test_range_upper_bound_inclusive(self, sl):
        _populate(sl, [(1, "a"), (2, "b"), (3, "c")])
        assert sl.range_query(2, 3) == [(2, "b"), (3, "c")]

    def test_range_full_span(self, sl):
        pairs = [(i, str(i)) for i in range(5)]
        _populate(sl, pairs)
        assert sl.range_query(-100, 100) == pairs

    def test_range_single_key(self, sl):
        _populate(sl, [(1, "a"), (5, "e"), (9, "i")])
        assert sl.range_query(5, 5) == [(5, "e")]

    def test_range_empty_when_no_keys_in_range(self, sl):
        _populate(sl, [(1, "a"), (10, "j")])
        assert sl.range_query(3, 7) == []

    def test_range_empty_on_empty_list(self, sl):
        assert sl.range_query(0, 100) == []

    def test_range_lo_greater_than_hi_is_empty(self, sl):
        _populate(sl, [(1, "a"), (2, "b"), (3, "c")])
        assert sl.range_query(5, 1) == []

    def test_range_below_all_keys(self, sl):
        _populate(sl, [(10, "a"), (20, "b")])
        assert sl.range_query(0, 5) == []

    def test_range_above_all_keys(self, sl):
        _populate(sl, [(10, "a"), (20, "b")])
        assert sl.range_query(30, 40) == []

    def test_range_results_sorted(self, sl):
        _populate(sl, [(7, "g"), (3, "c"), (5, "e"), (1, "a"), (9, "i")])
        result = sl.range_query(2, 8)
        keys = [k for k, _ in result]
        assert keys == sorted(keys)
        assert keys == [3, 5, 7]

    def test_range_with_negative_bounds(self, sl):
        _populate(sl, [(-3, "a"), (-1, "b"), (0, "c"), (2, "d")])
        assert sl.range_query(-2, 1) == [(-1, "b"), (0, "c")]


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
class TestEdgeCases:
    def test_search_on_empty(self, sl):
        assert sl.search(0) is None

    def test_delete_on_empty(self, sl):
        assert sl.delete(0) is False

    def test_single_element_lifecycle(self, sl):
        sl.insert(42, "answer")
        assert len(sl) == 1
        assert sl.search(42) == "answer"
        assert 42 in sl
        assert list(sl) == [(42, "answer")]
        assert sl.range_query(42, 42) == [(42, "answer")]
        assert sl.delete(42) is True
        assert len(sl) == 0
        assert list(sl) == []

    def test_repr_is_string(self, sl):
        _populate(sl, [(1, "a"), (2, "b")])
        r = repr(sl)
        assert isinstance(r, str)
        assert r  # non-empty

    def test_repr_empty(self, sl):
        assert isinstance(repr(sl), str)

    def test_duplicate_delete_second_is_false(self, sl):
        sl.insert(1, "a")
        assert sl.delete(1) is True
        assert sl.delete(1) is False

    def test_reverse_insertion_order_sorted_output(self, sl):
        for k in range(50, 0, -1):
            sl.insert(k, k * 10)
        assert [k for k, _ in sl] == list(range(1, 51))
        assert len(sl) == 50

    def test_level_never_drops_below_one_after_deletes(self):
        # Deleting every element must not let the active height collapse
        # below 1; a level of 0 would break the descent loops and leave the
        # structure unusable for subsequent operations.
        sl = SkipList(max_level=8, p=0.5)
        for i in range(64):
            sl.insert(i, i)
        for i in range(64):
            assert sl.delete(i) is True
            assert sl.level >= 1
        assert len(sl) == 0
        # Structure remains usable after fully draining and is still correct.
        sl.insert(123, "alive")
        assert sl.search(123) == "alive"
        assert list(sl) == [(123, "alive")]

    def test_empty_after_drain_searches_safely(self):
        # An emptied list must still answer search/contains/range without
        # error and report level >= 1 (no degenerate zero-height state).
        sl = SkipList()
        sl.insert(1, "a")
        sl.delete(1)
        assert sl.level >= 1
        assert sl.search(1) is None
        assert 1 not in sl
        assert sl.range_query(-10, 10) == []


# --------------------------------------------------------------------------- #
# Probabilistic level behavior
# --------------------------------------------------------------------------- #
class TestProbabilisticBehavior:
    def test_level_never_exceeds_max_level(self):
        sl = SkipList(max_level=4, p=0.9)  # high p pushes toward the cap
        for i in range(500):
            sl.insert(i, i)
        assert sl.level <= sl.max_level
        # correctness preserved despite capped height
        assert all(sl.search(i) == i for i in range(500))

    def test_height_is_logarithmic(self):
        random.seed(1234)
        n = 4096
        sl = SkipList(max_level=32, p=0.5)
        for i in range(n):
            sl.insert(i, i)
        # Expected height ~ log_{1/p}(n) = log2(4096) = 12.
        # Generous ceiling guards against linear-height degeneration.
        assert sl.level <= 3 * math.log2(n)

    def test_random_level_within_bounds(self):
        sl = SkipList(max_level=16, p=0.5)
        # Sample many generated levels; all must be 1..max_level.
        levels = [sl._random_level() for _ in range(2000)]
        assert all(1 <= lv <= sl.max_level for lv in levels)
        # With p=0.5 a meaningful fraction should be > 1 (not always level 1).
        assert any(lv > 1 for lv in levels)

    def test_level_distribution_geometric(self):
        random.seed(7)
        sl = SkipList(max_level=32, p=0.5)
        levels = [sl._random_level() for _ in range(20000)]
        frac_gt1 = sum(lv > 1 for lv in levels) / len(levels)
        # P(level > 1) == p == 0.5; allow slack for randomness.
        assert 0.4 < frac_gt1 < 0.6


# --------------------------------------------------------------------------- #
# Stress / correctness under load
# --------------------------------------------------------------------------- #
class TestStress:
    def test_1000_insertions_correct(self):
        random.seed(99)
        sl = SkipList()
        keys = list(range(1000))
        random.shuffle(keys)
        for k in keys:
            sl.insert(k, k * 2)
        assert len(sl) == 1000
        for k in range(1000):
            assert sl.search(k) == k * 2
        assert [k for k, _ in sl] == list(range(1000))

    def test_mixed_operations_match_dict_oracle(self):
        random.seed(2024)
        sl = SkipList()
        oracle = {}
        for _ in range(5000):
            op = random.random()
            key = random.randint(0, 200)
            if op < 0.55:
                val = random.randint(0, 10**6)
                sl.insert(key, val)
                oracle[key] = val
            elif op < 0.85:
                expected = key in oracle
                got = sl.delete(key)
                assert got == expected
                oracle.pop(key, None)
            else:
                assert sl.search(key) == oracle.get(key)
            assert len(sl) == len(oracle)
        assert sorted(oracle.items()) == list(sl)

    def test_range_query_matches_oracle(self):
        random.seed(555)
        sl = SkipList()
        oracle = {}
        for _ in range(2000):
            k = random.randint(-500, 500)
            v = random.randint(0, 1000)
            sl.insert(k, v)
            oracle[k] = v
        for _ in range(50):
            lo = random.randint(-500, 500)
            hi = random.randint(lo, 500)
            expected = sorted(
                (k, val) for k, val in oracle.items() if lo <= k <= hi
            )
            assert sl.range_query(lo, hi) == expected
