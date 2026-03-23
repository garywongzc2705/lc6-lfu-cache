"""
Level 4 — Cache Warming and Snapshot/Restore

Add two production-quality features: bulk cache warming for
preloading known-hot data, and snapshot/restore for cache
persistence and migration.

── Part A: Cache Warming ────────────────────────────────────────────

New method:
  cache.warm(entries)
    — bulk insert a list of (key, value) or (key, value, ttl) tuples
    — entries are inserted in order; if capacity is exceeded, LFU
      eviction applies normally
    — frequencies from warming count the same as regular puts
    — returns a dict:
      {
        "inserted": int,    # entries successfully inserted
        "skipped":  int,    # entries skipped (key already existed)
        "evicted":  int,    # entries evicted due to capacity during warm
      }
    — existing keys are skipped (not updated) — warm() is for
      cold-start preloading, not updating

── Part B: Snapshot and Restore ─────────────────────────────────────

New methods:
  cache.snapshot()
    — return a serializable dict representing the current cache state:
      {
        "entries": [
          {
            "key":       ...,
            "value":     ...,
            "frequency": int,
            "expires_at": float | None,
          },
          ...
        ],
        "capacity": int,
        "stats":    dict,   # same format as stats()
      }
    — entries are sorted by frequency desc, LRU tiebreak (same as most_frequent)
    — expired entries are NOT included
    — in-flight TTL timestamps are preserved as absolute values (expires_at)

  cache.restore(snapshot)
    — restore cache state from a snapshot dict
    — replaces all current state (equivalent to clear() then re-inserting)
    — entries are re-inserted with their original frequencies
    — TTL entries with expires_at in the past are skipped (already expired)
    — stats are restored from the snapshot
    — callbacks (on_evict, on_expire) are NOT restored — they are
      registered externally and preserved across restore()
    — raises ValueError if snapshot format is invalid

Semantics:
  - snapshot() + restore() round-trip must produce identical get/frequency
    results for all non-expired keys
  - restore() must set frequencies correctly — a key restored with
    frequency=5 should have the same eviction priority as one that was
    accessed 5 times naturally
  - capacity is restored from snapshot
  - restore() does NOT trigger on_evict callbacks even if capacity changes
"""

import pytest
from solution import LFUCache


def make_clock(start=1000.0):
    t = [start]
    def clock(): return t[0]
    def advance(s): t[0] += s
    return clock, advance


# ── Part A: Cache Warming ─────────────────────────────────────────────────────

def test_warm_basic():
    c = LFUCache(capacity=5)
    result = c.warm([(1, 10), (2, 20), (3, 30)])
    assert result["inserted"] == 3
    assert result["skipped"] == 0
    assert result["evicted"] == 0
    assert c.get(1) == 10
    assert c.get(2) == 20
    assert c.get(3) == 30


def test_warm_skips_existing_keys():
    c = LFUCache(capacity=5)
    c.put(1, 10)
    result = c.warm([(1, 99), (2, 20)])
    assert result["skipped"] == 1
    assert c.get(1) == 10  # unchanged


def test_warm_with_ttl():
    clock, advance = make_clock()
    c = LFUCache(capacity=5)
    c.set_clock(clock)
    c.warm([(1, 10, 30)])
    advance(31)
    assert c.get(1) == -1


def test_warm_evicts_lfu_when_over_capacity():
    c = LFUCache(capacity=3)
    c.put(1, 10)
    c.put(2, 20)
    c.put(3, 30)
    c.get(2)
    c.get(3)
    result = c.warm([(4, 40), (5, 50)])
    assert result["evicted"] == 2
    assert result["inserted"] == 2
    assert c.get(1) == -1  # evicted (freq=1, LRU)


def test_warm_frequencies_count():
    c = LFUCache(capacity=3)
    c.warm([(1, 10), (2, 20), (3, 30)])
    c.get(1)
    c.get(1)  # freq=3
    c.put(4, 40)  # evicts 2 or 3 (freq=1), not 1 (freq=3)
    assert c.get(1) == 10


def test_warm_empty_entries():
    c = LFUCache(capacity=3)
    result = c.warm([])
    assert result["inserted"] == 0
    assert result["skipped"] == 0
    assert result["evicted"] == 0


# ── Part B: Snapshot / Restore ────────────────────────────────────────────────

def test_snapshot_basic():
    c = LFUCache(capacity=3)
    c.put(1, 10)
    c.put(2, 20)
    snap = c.snapshot()
    assert snap["capacity"] == 3
    assert len(snap["entries"]) == 2


def test_snapshot_entry_fields():
    c = LFUCache(capacity=3)
    c.put(1, 10)
    c.get(1)
    snap = c.snapshot()
    entry = next(e for e in snap["entries"] if e["key"] == 1)
    assert entry["value"] == 10
    assert entry["frequency"] == 2
    assert entry["expires_at"] is None


def test_snapshot_with_ttl():
    clock, _ = make_clock(start=1000.0)
    c = LFUCache(capacity=3)
    c.set_clock(clock)
    c.put(1, 10, ttl=60)
    snap = c.snapshot()
    entry = next(e for e in snap["entries"] if e["key"] == 1)
    assert entry["expires_at"] == pytest.approx(1060.0)


def test_snapshot_excludes_expired():
    clock, advance = make_clock()
    c = LFUCache(capacity=3)
    c.set_clock(clock)
    c.put(1, 10, ttl=5)
    c.put(2, 20)
    advance(6)
    snap = c.snapshot()
    keys = [e["key"] for e in snap["entries"]]
    assert 1 not in keys
    assert 2 in keys


def test_snapshot_sorted_by_frequency():
    c = LFUCache(capacity=5)
    c.put(1, 10)
    c.put(2, 20)
    c.put(3, 30)
    for _ in range(4): c.get(1)   # freq=5
    for _ in range(2): c.get(2)   # freq=3
    snap = c.snapshot()
    freqs = [e["frequency"] for e in snap["entries"]]
    assert freqs == sorted(freqs, reverse=True)


def test_snapshot_includes_stats():
    c = LFUCache(capacity=3)
    c.put(1, 10)
    c.get(1)
    snap = c.snapshot()
    assert snap["stats"]["hits"] == 1


# --- restore ---

def test_restore_basic_round_trip():
    c = LFUCache(capacity=3)
    c.put(1, 10)
    c.put(2, 20)
    c.get(1)
    snap = c.snapshot()

    c2 = LFUCache(capacity=1)
    c2.restore(snap)
    assert c2.get(1) == 10
    assert c2.get(2) == 20


def test_restore_preserves_frequencies():
    c = LFUCache(capacity=3)
    c.put(1, 10)
    c.put(2, 20)
    c.put(3, 30)
    for _ in range(4): c.get(1)   # freq=5
    for _ in range(2): c.get(2)   # freq=3
    snap = c.snapshot()

    c2 = LFUCache(capacity=3)
    c2.restore(snap)
    assert c2.frequency(1) == 5
    assert c2.frequency(2) == 3
    assert c2.frequency(3) == 1


def test_restore_eviction_order_correct():
    c = LFUCache(capacity=3)
    c.put(1, 10)
    c.put(2, 20)
    c.put(3, 30)
    c.get(1)
    c.get(1)   # freq=3
    c.get(2)   # freq=2
    snap = c.snapshot()

    c2 = LFUCache(capacity=3)
    c2.restore(snap)
    c2.put(4, 40)   # evicts 3 (freq=1)
    assert c2.get(3) == -1
    assert c2.get(1) == 10


def test_restore_skips_expired_entries():
    clock, advance = make_clock(start=1000.0)
    c = LFUCache(capacity=3)
    c.set_clock(clock)
    c.put(1, 10, ttl=30)
    c.put(2, 20)
    snap = c.snapshot()

    advance(31)
    c2 = LFUCache(capacity=3)
    c2.set_clock(clock)
    c2.restore(snap)
    assert c2.get(1) == -1   # expired by the time of restore
    assert c2.get(2) == 20


def test_restore_preserves_capacity():
    c = LFUCache(capacity=10)
    c.put(1, 10)
    snap = c.snapshot()

    c2 = LFUCache(capacity=1)
    c2.restore(snap)
    assert c2.capacity == 10


def test_restore_preserves_callbacks():
    evicted = []
    c = LFUCache(capacity=2)
    c.on_evict(lambda k, v: evicted.append(k))
    c.put(1, 10)
    snap = c.snapshot()

    c.restore(snap)  # callbacks should still be registered
    c.put(2, 20)
    c.put(3, 30)    # evicts 1
    assert 1 in evicted


def test_restore_invalid_snapshot_raises():
    c = LFUCache(capacity=3)
    with pytest.raises(ValueError):
        c.restore({"bad": "format"})


def test_restore_clears_existing_state():
    c = LFUCache(capacity=3)
    c.put(99, 999)
    snap = {"entries": [], "capacity": 3, "stats": {
        "hits": 0, "misses": 0, "evictions": 0, "expirations": 0, "hit_rate": 0.0
    }}
    c.restore(snap)
    assert c.get(99) == -1
    assert c.size() == 0
