"""
Level 3 — Eviction Callbacks and Frequency Inspection

Add the same callback hooks as LRU Level 3, plus frequency
inspection methods that are unique to LFU.

New methods:
  cache.on_evict(fn)
    — register a callback: fn(key, value)
    — called when an item is evicted due to capacity overflow
    — multiple callbacks accumulate, all called in order

  cache.on_expire(fn)
    — register a callback: fn(key, value)
    — called when an item is lazily removed due to TTL expiry
    — multiple callbacks accumulate

  cache.remove(key)
    — explicitly remove a key
    — returns value if existed, else -1
    — does NOT trigger on_evict or on_expire
    — does not count as eviction or expiration in stats()

  cache.frequency(key)
    — return the current access frequency of key
    — returns 0 if key is not in cache or has expired

  cache.most_frequent()
    — return a list of (key, frequency) tuples for the top-5
      most frequently accessed keys, sorted by frequency desc
    — ties broken by most recently used (most recent first)
    — returns fewer than 5 if cache has fewer than 5 items
    — expired items are excluded

  cache.least_frequent()
    — return the (key, frequency) tuple for the item that would
      be evicted next (the LFU candidate)
    — returns None if cache is empty
    — expired items are excluded

Semantics:
  - on_evict fires for capacity-overflow evictions only
  - on_expire fires for TTL expiry only (lazy, at access time)
  - remove() on a missing or expired key returns -1 silently
  - clear() resets callbacks too (consistent with LRU behavior)
  - frequency() does NOT increment the key's frequency
  - all previous tests must pass
"""

import pytest
from solution import LFUCache


def make_clock(start=0.0):
    t = [start]
    def clock(): return t[0]
    def advance(s): t[0] += s
    return clock, advance


# --- on_evict ---

def test_on_evict_called_on_overflow():
    evicted = []
    c = LFUCache(capacity=2)
    c.on_evict(lambda k, v: evicted.append((k, v)))
    c.put(1, 10)
    c.put(2, 20)
    c.get(2)       # freq 2→2
    c.put(3, 30)   # evicts 1 (freq=1)
    assert evicted == [(1, 10)]


def test_on_evict_multiple_callbacks():
    log1, log2 = [], []
    c = LFUCache(capacity=1)
    c.on_evict(lambda k, v: log1.append(k))
    c.on_evict(lambda k, v: log2.append(v))
    c.put(1, 10)
    c.put(2, 20)
    assert log1 == [1]
    assert log2 == [10]


def test_on_evict_not_triggered_by_remove():
    evicted = []
    c = LFUCache(capacity=2)
    c.on_evict(lambda k, v: evicted.append(k))
    c.put(1, 10)
    c.remove(1)
    assert evicted == []


def test_on_evict_not_triggered_by_clear():
    evicted = []
    c = LFUCache(capacity=2)
    c.on_evict(lambda k, v: evicted.append(k))
    c.put(1, 10)
    c.put(2, 20)
    c.clear()
    assert evicted == []


# --- on_expire ---

def test_on_expire_called_on_ttl_expiry():
    expired = []
    clock, advance = make_clock()
    c = LFUCache(capacity=3)
    c.set_clock(clock)
    c.on_expire(lambda k, v: expired.append((k, v)))
    c.put(1, 10, ttl=5)
    advance(6)
    c.get(1)
    assert expired == [(1, 10)]


def test_on_expire_not_called_before_expiry():
    expired = []
    clock, advance = make_clock()
    c = LFUCache(capacity=3)
    c.set_clock(clock)
    c.on_expire(lambda k, v: expired.append(k))
    c.put(1, 10, ttl=5)
    advance(3)
    c.get(1)
    assert expired == []


def test_on_expire_not_triggered_by_eviction():
    expired = []
    c = LFUCache(capacity=2)
    c.on_expire(lambda k, v: expired.append(k))
    c.put(1, 10)
    c.put(2, 20)
    c.put(3, 30)
    assert expired == []


def test_on_expire_multiple_callbacks():
    log1, log2 = [], []
    clock, advance = make_clock()
    c = LFUCache(capacity=3)
    c.set_clock(clock)
    c.on_expire(lambda k, v: log1.append(k))
    c.on_expire(lambda k, v: log2.append(v))
    c.put(1, 42, ttl=5)
    advance(6)
    c.get(1)
    assert log1 == [1]
    assert log2 == [42]


# --- remove() ---

def test_remove_existing_key():
    c = LFUCache(capacity=3)
    c.put(1, 10)
    assert c.remove(1) == 10
    assert c.get(1) == -1


def test_remove_missing_key():
    c = LFUCache(capacity=3)
    assert c.remove(99) == -1


def test_remove_decrements_size():
    c = LFUCache(capacity=3)
    c.put(1, 10)
    c.put(2, 20)
    c.remove(1)
    assert c.size() == 1


def test_remove_not_counted_as_eviction():
    c = LFUCache(capacity=3)
    c.put(1, 10)
    c.remove(1)
    assert c.stats()["evictions"] == 0


def test_remove_cleans_up_frequency():
    c = LFUCache(capacity=2)
    c.put(1, 10)
    c.get(1)
    c.get(1)   # freq=3
    c.remove(1)
    c.put(1, 99)
    c.put(2, 20)
    c.put(3, 30)   # evicts 1 (freq=1 after remove+reinsert)
    assert c.get(1) == -1


# --- frequency() ---

def test_frequency_after_put():
    c = LFUCache(capacity=3)
    c.put(1, 10)
    assert c.frequency(1) == 1


def test_frequency_increments_on_get():
    c = LFUCache(capacity=3)
    c.put(1, 10)
    c.get(1)
    c.get(1)
    assert c.frequency(1) == 3


def test_frequency_missing_key():
    c = LFUCache(capacity=3)
    assert c.frequency(99) == 0


def test_frequency_does_not_increment():
    c = LFUCache(capacity=2)
    c.put(1, 10)
    c.put(2, 20)
    c.frequency(1)  # should NOT increment freq of 1
    c.frequency(1)
    c.frequency(1)
    c.put(3, 30)    # 1 should be evicted (still freq=1)
    assert c.get(1) == -1


def test_frequency_expired_key():
    clock, advance = make_clock()
    c = LFUCache(capacity=3)
    c.set_clock(clock)
    c.put(1, 10, ttl=5)
    advance(6)
    assert c.frequency(1) == 0


# --- most_frequent() ---

def test_most_frequent_basic():
    c = LFUCache(capacity=5)
    c.put(1, 10)
    c.put(2, 20)
    c.put(3, 30)
    for _ in range(5): c.get(1)   # freq=6
    for _ in range(3): c.get(2)   # freq=4
    for _ in range(1): c.get(3)   # freq=2
    result = c.most_frequent()
    assert result[0] == (1, 6)
    assert result[1] == (2, 4)
    assert result[2] == (3, 2)


def test_most_frequent_max_five():
    c = LFUCache(capacity=10)
    for i in range(8):
        c.put(i, i)
        for _ in range(i): c.get(i)
    result = c.most_frequent()
    assert len(result) == 5


def test_most_frequent_fewer_than_five():
    c = LFUCache(capacity=3)
    c.put(1, 10)
    c.put(2, 20)
    result = c.most_frequent()
    assert len(result) == 2


# --- least_frequent() ---

def test_least_frequent_basic():
    c = LFUCache(capacity=3)
    c.put(1, 10)
    c.put(2, 20)
    c.put(3, 30)
    c.get(1)
    c.get(1)   # freq=3
    c.get(2)   # freq=2
    # 3 has freq=1, is LFU candidate
    result = c.least_frequent()
    assert result == (3, 1)


def test_least_frequent_empty():
    c = LFUCache(capacity=3)
    assert c.least_frequent() is None


def test_least_frequent_tie_broken_by_lru():
    c = LFUCache(capacity=3)
    c.put(1, 10)   # freq=1, inserted first
    c.put(2, 20)   # freq=1, inserted second
    c.put(3, 30)   # freq=1, inserted third
    result = c.least_frequent()
    assert result == (1, 1)   # 1 is LRU among freq=1
