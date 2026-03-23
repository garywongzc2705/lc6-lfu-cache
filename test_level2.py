"""
Level 2 — TTL and Metrics

Add time-based expiry and observability to your LFU cache.
Same semantics as the LRU cache TTL/metrics you built earlier.

New methods:
  cache.put(key, value, ttl=None)
    — ttl is optional seconds until expiry
    — if ttl=None, the entry never expires
    — expired entries behave as if they don't exist
    — expiry is checked lazily at read time

  cache.set_clock(fn)
    — inject a clock function (returns float)
    — defaults to time.time

  cache.stats()
    — return a dict:
      {
        "hits":        int,
        "misses":      int,
        "evictions":   int,
        "expirations": int,
        "hit_rate":    float,
      }

  cache.clear()
    — remove all entries, reset stats

Semantics:
  - An expired item is removed from the cache on access (lazy cleanup)
  - Expiry counts as an expiration, not an eviction
  - put() on an already-expired key replaces it (no expiration counted)
  - ttl is relative to the time of put()
  - Updating an existing key with put() resets its TTL
  - contains() returning False for expired keys does NOT count as a miss
  - size() never counts expired keys
  - frequency is NOT incremented when accessing an expired key
  - expired entries removed lazily do not fire eviction callbacks
    (no callbacks in this level — that's Level 3)
"""

import pytest
import time
from solution import LFUCache


def make_clock(start=0.0):
    t = [start]
    def clock(): return t[0]
    def advance(s): t[0] += s
    return clock, advance


@pytest.fixture
def cache():
    clock, _ = make_clock()
    c = LFUCache(capacity=3)
    c.set_clock(clock)
    return c


# --- Basic TTL ---

def test_ttl_readable_before_expiry():
    clock, advance = make_clock()
    c = LFUCache(capacity=3)
    c.set_clock(clock)
    c.put(1, 10, ttl=30)
    advance(15)
    assert c.get(1) == 10


def test_ttl_expires():
    clock, advance = make_clock()
    c = LFUCache(capacity=3)
    c.set_clock(clock)
    c.put(1, 10, ttl=30)
    advance(30)
    assert c.get(1) == -1


def test_ttl_expires_strictly():
    clock, advance = make_clock()
    c = LFUCache(capacity=3)
    c.set_clock(clock)
    c.put(1, 10, ttl=30)
    advance(30.001)
    assert c.get(1) == -1


def test_no_ttl_never_expires():
    clock, advance = make_clock()
    c = LFUCache(capacity=3)
    c.set_clock(clock)
    c.put(1, 10)
    advance(9999)
    assert c.get(1) == 10


def test_put_resets_ttl():
    clock, advance = make_clock()
    c = LFUCache(capacity=3)
    c.set_clock(clock)
    c.put(1, 10, ttl=20)
    advance(15)
    c.put(1, 99, ttl=20)
    advance(10)
    assert c.get(1) == 99


def test_contains_false_for_expired():
    clock, advance = make_clock()
    c = LFUCache(capacity=3)
    c.set_clock(clock)
    c.put(1, 10, ttl=5)
    advance(6)
    assert c.contains(1) is False


def test_size_excludes_expired():
    clock, advance = make_clock()
    c = LFUCache(capacity=3)
    c.set_clock(clock)
    c.put(1, 10, ttl=5)
    c.put(2, 20)
    advance(6)
    assert c.size() == 1


def test_expired_key_replaced_by_put():
    clock, advance = make_clock()
    c = LFUCache(capacity=3)
    c.set_clock(clock)
    c.put(1, 10, ttl=5)
    advance(6)
    c.put(1, 99)
    assert c.get(1) == 99


def test_expired_key_frequency_reset():
    clock, advance = make_clock()
    c = LFUCache(capacity=2)
    c.set_clock(clock)
    c.put(1, 10, ttl=5)
    c.get(1)
    c.get(1)
    c.get(1)   # freq=4 before expiry
    advance(6)
    c.put(1, 99)  # re-insert after expiry — freq should reset to 1
    c.put(2, 20)
    c.put(3, 30)  # evicts 1 (freq=1) not 2 (freq=1, inserted after 1)
    assert c.get(1) == -1
    assert c.get(2) == 20


# --- Metrics ---

def test_stats_initial(cache):
    s = cache.stats()
    assert s["hits"] == 0
    assert s["misses"] == 0
    assert s["evictions"] == 0
    assert s["expirations"] == 0
    assert s["hit_rate"] == pytest.approx(0.0)


def test_stats_hits(cache):
    cache.put(1, 10)
    cache.get(1)
    cache.get(1)
    assert cache.stats()["hits"] == 2


def test_stats_misses(cache):
    cache.get(99)
    cache.get(98)
    assert cache.stats()["misses"] == 2


def test_stats_hit_rate(cache):
    cache.put(1, 10)
    cache.get(1)
    cache.get(99)
    assert cache.stats()["hit_rate"] == pytest.approx(0.5)


def test_stats_evictions(cache):
    cache.put(1, 10)
    cache.put(2, 20)
    cache.put(3, 30)
    cache.put(4, 40)
    assert cache.stats()["evictions"] == 1


def test_stats_expirations():
    clock, advance = make_clock()
    c = LFUCache(capacity=3)
    c.set_clock(clock)
    c.put(1, 10, ttl=5)
    advance(6)
    c.get(1)
    assert c.stats()["expirations"] == 1


def test_stats_expiration_not_eviction():
    clock, advance = make_clock()
    c = LFUCache(capacity=3)
    c.set_clock(clock)
    c.put(1, 10, ttl=5)
    advance(6)
    c.get(1)
    s = c.stats()
    assert s["expirations"] == 1
    assert s["evictions"] == 0


def test_stats_expired_not_counted_as_miss():
    clock, advance = make_clock()
    c = LFUCache(capacity=3)
    c.set_clock(clock)
    c.put(1, 10, ttl=5)
    advance(6)
    c.get(1)
    s = c.stats()
    assert s["misses"] == 0
    assert s["expirations"] == 1


def test_stats_hit_rate_zero_gets(cache):
    assert cache.stats()["hit_rate"] == pytest.approx(0.0)


# --- clear() ---

def test_clear_removes_all(cache):
    cache.put(1, 10)
    cache.put(2, 20)
    cache.clear()
    assert cache.size() == 0
    assert cache.get(1) == -1


def test_clear_resets_stats(cache):
    cache.put(1, 10)
    cache.get(1)
    cache.get(99)
    cache.clear()
    s = cache.stats()
    assert s["hits"] == 0
    assert s["misses"] == 0
    assert s["evictions"] == 0


def test_clear_resets_frequencies(cache):
    cache.put(1, 10)
    cache.get(1)
    cache.get(1)
    cache.get(1)  # freq=4
    cache.clear()
    cache.put(1, 10)
    cache.put(2, 20)
    cache.put(3, 30)
    cache.put(4, 40)  # 1 should be evicted (freq=1 after clear)
    assert cache.get(1) == -1
