"""
Level 1 — LFU Cache

Implement an LFU (Least Frequently Used) cache with a fixed capacity.

Constructor:
  LFUCache(capacity)

Methods:
  cache.get(key)
    — return the value for key if present, else -1
    — increments the key's access frequency

  cache.put(key, value)
    — insert or update the key-value pair
    — increments the key's access frequency
    — if at capacity and key is new, evict the least frequently used item
    — ties in frequency are broken by LRU order:
      among items with equal frequency, evict the one accessed
      least recently

  cache.size()
    — return current number of items in the cache

  cache.contains(key)
    — return True if key is in cache, False otherwise
    — does NOT increment frequency

Semantics:
  - Frequency starts at 1 when a key is first inserted via put()
  - get() and put() both increment frequency
  - put() on an existing key updates value AND increments frequency
  - Eviction removes exactly one item per put() that overflows capacity
  - capacity=1 is valid
  - get() on a missing key returns -1 and does NOT affect any frequencies

Performance requirement (will be asked about, not tested):
  - get() must be O(1)
  - put() must be O(1)
"""

import pytest
from solution import LFUCache


@pytest.fixture
def cache():
    return LFUCache(capacity=3)


# --- Basic get/put ---

def test_put_and_get(cache):
    cache.put(1, 10)
    assert cache.get(1) == 10


def test_get_missing_returns_minus_one(cache):
    assert cache.get(99) == -1


def test_put_updates_existing(cache):
    cache.put(1, 10)
    cache.put(1, 99)
    assert cache.get(1) == 99


def test_multiple_keys(cache):
    cache.put(1, 10)
    cache.put(2, 20)
    cache.put(3, 30)
    assert cache.get(1) == 10
    assert cache.get(2) == 20
    assert cache.get(3) == 30


# --- size and contains ---

def test_size_empty(cache):
    assert cache.size() == 0


def test_size_after_puts(cache):
    cache.put(1, 10)
    cache.put(2, 20)
    assert cache.size() == 2


def test_size_does_not_exceed_capacity(cache):
    for i in range(10):
        cache.put(i, i)
    assert cache.size() == 3


def test_size_update_does_not_increase(cache):
    cache.put(1, 10)
    cache.put(1, 20)
    assert cache.size() == 1


def test_contains_existing(cache):
    cache.put(1, 10)
    assert cache.contains(1) is True


def test_contains_missing(cache):
    assert cache.contains(99) is False


def test_contains_does_not_affect_frequency(cache):
    cache.put(1, 10)
    cache.put(2, 20)
    cache.put(3, 30)
    cache.contains(1)  # should NOT increment frequency of 1
    cache.put(4, 40)   # 1 should be evicted (freq=1, LRU among freq=1)
    assert cache.get(1) == -1


# --- LFU eviction ---

def test_evicts_lfu_on_overflow(cache):
    cache.put(1, 10)
    cache.put(2, 20)
    cache.put(3, 30)
    cache.get(1)   # freq 1→2
    cache.get(1)   # freq 1→3
    cache.get(2)   # freq 2→2
    cache.put(4, 40)  # evicts 3 (freq=1)
    assert cache.get(3) == -1
    assert cache.get(4) == 40


def test_frequency_tie_broken_by_lru():
    c = LFUCache(capacity=3)
    c.put(1, 10)   # freq=1
    c.put(2, 20)   # freq=1
    c.put(3, 30)   # freq=1
    # all freq=1, 1 is oldest (LRU)
    c.put(4, 40)   # evicts 1
    assert c.get(1) == -1
    assert c.get(2) == 20
    assert c.get(3) == 30
    assert c.get(4) == 40


def test_get_increments_frequency():
    c = LFUCache(capacity=2)
    c.put(1, 10)   # freq=1
    c.put(2, 20)   # freq=1
    c.get(1)       # freq 1→2
    c.put(3, 30)   # evicts 2 (freq=1), not 1 (freq=2)
    assert c.get(1) == 10
    assert c.get(2) == -1
    assert c.get(3) == 30


def test_put_update_increments_frequency():
    c = LFUCache(capacity=2)
    c.put(1, 10)   # freq=1
    c.put(2, 20)   # freq=1
    c.put(1, 99)   # freq 1→2, value updated
    c.put(3, 30)   # evicts 2 (freq=1)
    assert c.get(1) == 99
    assert c.get(2) == -1


def test_capacity_one():
    c = LFUCache(capacity=1)
    c.put(1, 10)
    c.put(2, 20)
    assert c.get(1) == -1
    assert c.get(2) == 20


def test_eviction_respects_frequency_not_insertion_order():
    c = LFUCache(capacity=3)
    c.put(1, 10)
    c.put(2, 20)
    c.put(3, 30)
    c.get(3)       # freq 3→2
    c.get(3)       # freq 3→3
    c.get(2)       # freq 2→2
    c.get(1)       # freq 1→2
    c.put(4, 40)   # all freq=2, evict oldest: 3 was accessed most recently
    # order of freq=2 accesses: put(1),put(2),put(3) → get(3),get(2),get(1)
    # LRU among freq=2: 2 was accessed before 1 in the get sequence
    # After gets: 3(freq=3), 2(freq=2,last get), 1(freq=2, last get after 2)
    # min_freq=2, LRU among freq=2 is 2
    assert c.get(2) == -1
    assert c.get(1) == 10
    assert c.get(3) == 30
