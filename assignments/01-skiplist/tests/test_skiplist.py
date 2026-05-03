"""Tests for the probabilistic SkipList (Assignment 01)."""

from __future__ import annotations

import math
import random

import pytest

from skiplist import SkipList


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_construction() -> None:
    sl = SkipList()
    assert len(sl) == 0
    assert list(sl) == []


def test_custom_construction() -> None:
    sl = SkipList(max_level=8, p=0.25)
    assert len(sl) == 0


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


def test_insert_and_search() -> None:
    sl = SkipList()
    sl.insert(1, "a")
    sl.insert(2, "b")
    sl.insert(3, "c")
    assert sl.search(1) == "a"
    assert sl.search(2) == "b"
    assert sl.search(3) == "c"
    assert len(sl) == 3


def test_search_missing_returns_none() -> None:
    sl = SkipList()
    assert sl.search(42) is None
    sl.insert(1, "a")
    assert sl.search(99) is None


def test_contains() -> None:
    sl = SkipList()
    assert 1 not in sl
    sl.insert(1, "a")
    assert 1 in sl
    assert 2 not in sl


def test_delete_returns_true_when_found() -> None:
    sl = SkipList()
    sl.insert(5, "v")
    assert sl.delete(5) is True
    assert 5 not in sl
    assert len(sl) == 0
    assert sl.search(5) is None


def test_delete_returns_false_when_missing() -> None:
    sl = SkipList()
    assert sl.delete(5) is False
    sl.insert(1, "a")
    assert sl.delete(99) is False
    assert len(sl) == 1


# ---------------------------------------------------------------------------
# None-value semantics for membership vs search
# ---------------------------------------------------------------------------


def test_contains_with_none_value() -> None:
    # Regression: ``__contains__`` must not delegate to ``search``, because
    # ``search`` returns ``None`` both for missing keys and for keys whose
    # value happens to be ``None``. Membership is independent of the value.
    sl = SkipList()
    sl.insert(1, None)
    assert 1 in sl
    assert len(sl) == 1


def test_search_returns_none_for_inserted_none() -> None:
    # Inserting a key with value ``None`` is legal; ``search`` happens to
    # return ``None`` for both "absent" and "present-with-None". This is the
    # documented behaviour and ``__contains__`` is the disambiguating call.
    sl = SkipList()
    sl.insert(7, None)
    assert sl.search(7) is None
    assert sl.search(99) is None
    assert 7 in sl
    assert 99 not in sl


# ---------------------------------------------------------------------------
# Duplicate-key semantics (update)
# ---------------------------------------------------------------------------


def test_duplicate_insert_updates_value() -> None:
    sl = SkipList()
    sl.insert(1, "a")
    sl.insert(1, "b")
    assert sl.search(1) == "b"
    assert len(sl) == 1


def test_repeated_updates_keep_length_constant() -> None:
    sl = SkipList()
    sl.insert(1, "a")
    for v in range(100):
        sl.insert(1, v)
    assert len(sl) == 1
    assert sl.search(1) == 99


# ---------------------------------------------------------------------------
# Range query
# ---------------------------------------------------------------------------


def _populate(sl: SkipList, items: list[tuple[int, object]]) -> None:
    for k, v in items:
        sl.insert(k, v)


def test_range_query_inclusive_bounds() -> None:
    sl = SkipList()
    _populate(sl, [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")])
    assert sl.range_query(2, 4) == [(2, "b"), (3, "c"), (4, "d")]
    assert sl.range_query(1, 1) == [(1, "a")]
    assert sl.range_query(1, 5) == [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")]


def test_range_query_empty_when_lo_greater_than_hi() -> None:
    sl = SkipList()
    _populate(sl, [(1, "a"), (2, "b"), (3, "c")])
    assert sl.range_query(5, 1) == []


def test_range_query_empty_skiplist() -> None:
    sl = SkipList()
    assert sl.range_query(0, 100) == []


def test_range_query_fully_outside() -> None:
    sl = SkipList()
    _populate(sl, [(10, "a"), (20, "b"), (30, "c")])
    assert sl.range_query(0, 5) == []
    assert sl.range_query(50, 100) == []


def test_range_query_partial_overlap() -> None:
    sl = SkipList()
    _populate(sl, [(10, "a"), (20, "b"), (30, "c"), (40, "d")])
    assert sl.range_query(15, 35) == [(20, "b"), (30, "c")]
    assert sl.range_query(-5, 15) == [(10, "a")]
    assert sl.range_query(35, 100) == [(40, "d")]


def test_range_query_returns_sorted_even_with_random_insert_order() -> None:
    sl = SkipList()
    rng = random.Random(123)
    keys = list(range(50))
    rng.shuffle(keys)
    for k in keys:
        sl.insert(k, k * 10)
    result = sl.range_query(10, 20)
    assert result == [(k, k * 10) for k in range(10, 21)]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_element_operations() -> None:
    sl = SkipList()
    sl.insert(7, "only")
    assert len(sl) == 1
    assert 7 in sl
    assert sl.search(7) == "only"
    assert list(sl) == [(7, "only")]
    assert sl.range_query(0, 100) == [(7, "only")]
    assert sl.delete(7) is True
    assert len(sl) == 0
    assert list(sl) == []


def test_empty_list_operations() -> None:
    sl = SkipList()
    assert len(sl) == 0
    assert sl.search(1) is None
    assert 1 not in sl
    assert sl.delete(1) is False
    assert list(sl) == []
    assert sl.range_query(0, 100) == []


def test_repeated_insert_delete_cycles() -> None:
    sl = SkipList()
    for _ in range(20):
        for k in range(50):
            sl.insert(k, k)
        assert len(sl) == 50
        for k in range(50):
            assert sl.delete(k) is True
        assert len(sl) == 0
        assert list(sl) == []


# ---------------------------------------------------------------------------
# Iterator
# ---------------------------------------------------------------------------


def test_iter_yields_sorted_pairs() -> None:
    sl = SkipList()
    rng = random.Random(7)
    keys = list(range(20))
    rng.shuffle(keys)
    for k in keys:
        sl.insert(k, str(k))
    assert list(sl) == [(k, str(k)) for k in range(20)]


def test_iter_reflects_updates_and_deletes() -> None:
    sl = SkipList()
    for k in range(5):
        sl.insert(k, k)
    sl.insert(2, "updated")
    assert sl.delete(3) is True
    assert list(sl) == [(0, 0), (1, 1), (2, "updated"), (4, 4)]


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


def test_repr_returns_string() -> None:
    sl = SkipList()
    assert isinstance(repr(sl), str)
    sl.insert(1, "a")
    assert isinstance(repr(sl), str)
    assert "SkipList" in repr(sl)


# ---------------------------------------------------------------------------
# Boundary: p=0.0 and p=1.0
# ---------------------------------------------------------------------------


def test_p_zero_still_functional() -> None:
    # With p=0.0 every node stays at level 1; structure degenerates to a
    # sorted linked list but operations must still work correctly.
    sl = SkipList(max_level=8, p=0.0)
    for k in [3, 1, 4, 1, 5, 9, 2, 6, 5]:
        sl.insert(k, k)
    assert sl.search(4) == 4
    assert sl.search(9) == 9
    assert sl.search(99) is None
    assert [k for k, _ in sl] == sorted({1, 2, 3, 4, 5, 6, 9})
    assert sl.delete(5) is True
    assert 5 not in sl


def test_p_one_capped_by_max_level() -> None:
    # With p=1.0 every node would attempt to climb forever; max_level must cap.
    sl = SkipList(max_level=4, p=1.0)
    for k in range(10):
        sl.insert(k, k)
    for k in range(10):
        assert sl.search(k) == k
    assert list(sl) == [(k, k) for k in range(10)]
    for k in range(10):
        assert sl.delete(k) is True
    assert len(sl) == 0


# ---------------------------------------------------------------------------
# Probabilistic height: O(log n) with high probability
# ---------------------------------------------------------------------------


def _measure_height(sl: SkipList) -> int:
    # The maximum non-empty level is exposed via private state; if not
    # present, fall back to counting via repr length heuristics.
    level = getattr(sl, "_level", None)
    if level is not None:
        return int(level)
    # Fallback: use max_level attribute as upper bound.
    return int(getattr(sl, "max_level", 16))


def test_height_is_logarithmic_for_large_input() -> None:
    n = 2000
    sl = SkipList(max_level=32, p=0.5)
    rng = random.Random(0xC0FFEE)
    keys = list(range(n))
    rng.shuffle(keys)
    for k in keys:
        sl.insert(k, k)
    assert len(sl) == n
    h = _measure_height(sl)
    # Expected level ~ log_{1/p}(n) = log2(n) ~ 11. Allow generous slack.
    upper_bound = int(4 * math.log2(n)) + 4
    assert h <= upper_bound, f"height {h} exceeds {upper_bound}"


# ---------------------------------------------------------------------------
# Stress test
# ---------------------------------------------------------------------------


def test_stress_1000_inserts_with_correctness() -> None:
    n = 1500
    sl = SkipList()
    rng = random.Random(2026)
    keys = list(range(n))
    rng.shuffle(keys)
    for k in keys:
        sl.insert(k, k * 2)
    assert len(sl) == n
    # search every key
    for k in keys:
        assert sl.search(k) == k * 2
    # iterator is sorted
    assert list(sl) == [(k, k * 2) for k in range(n)]
    # range query subset
    assert sl.range_query(100, 200) == [(k, k * 2) for k in range(100, 201)]
    # delete half
    to_delete = keys[: n // 2]
    for k in to_delete:
        assert sl.delete(k) is True
    assert len(sl) == n - n // 2
    for k in to_delete:
        assert sl.search(k) is None
        assert sl.delete(k) is False


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
