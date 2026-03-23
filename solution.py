from collections import OrderedDict, defaultdict
import heapq
import time

# Anthropic — Staff Engineer · Incremental Coding Round #7
# solution.py
#
# Write your implementation here.
# You may add any classes, functions, or methods you need.
# Do not import external libraries — stdlib only.
#
# Before each level, add a comment block explaining your approach:
#
# --- LEVEL 1 APPROACH ---
# (your design notes here)
# -------------------------


class CacheEntry:
    def __init__(self, key, val, ttl=None):
        self.key = key
        self.val = val
        self.ttl = ttl


class LFUCache:
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.clock = time.time
        self._reset()

    def _reset(self):
        self.cache = {}
        self.key_freq = defaultdict(int)
        self.freq_count = defaultdict(OrderedDict)
        self.min_freq = 1
        self.ttl_entries = []
        self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0, "expirations": 0}

    def get(self, key):
        is_key_expired = self._sync_ttl_items(key)

        if key in self.cache:
            self.cache_stats["hits"] += 1
            entry = self.cache[key]
            self._update_freqency(entry)
            return entry.val

        if not is_key_expired:
            self.cache_stats["misses"] += 1
        return -1

    def put(self, key, value, ttl=None):
        self._sync_ttl_items()
        if key not in self.cache and len(self.cache) == self.capacity:
            self._evict()

        if key in self.cache:
            self._update_ttl(self.cache[key])

        entry = self.cache[key] if key in self.cache else CacheEntry(key, value, ttl)
        self._update_freqency(entry)
        entry.val = value
        self.cache[key] = entry

        if ttl:
            heapq.heappush(self.ttl_entries, (self.clock() + ttl, entry))

    def size(self):
        self._sync_ttl_items()
        return len(self.cache)

    def contains(self, key):
        self._sync_ttl_items()
        return key in self.cache

    def _update_ttl(self, target: CacheEntry):
        self.ttl_entries = [
            entry for entry in self.ttl_entries if entry[1].key != target.key
        ]
        heapq.heapify(self.ttl_entries)

    def _update_freqency(self, entry: CacheEntry):
        old_freq = self.key_freq[entry.key]
        new_freq = old_freq + 1
        self.freq_count[new_freq][entry.key] = entry.key
        self.key_freq[entry.key] = new_freq

        if entry.key in self.freq_count[old_freq]:
            del self.freq_count[old_freq][entry.key]

            if len(self.freq_count) == 0:
                self.min_freq = 1

        if old_freq == self.min_freq and len(self.freq_count[old_freq]) == 0:
            self.min_freq = old_freq + 1

    def _evict(self):
        entry_tuple = self.freq_count[self.min_freq].popitem(last=False)
        key = entry_tuple[0]
        del self.cache[key]
        del self.key_freq[key]
        self.cache_stats["evictions"] += 1

    def set_clock(self, fn):
        self.clock = fn

    def stats(self):
        hits = self.cache_stats["hits"]
        misses = self.cache_stats["misses"]
        evictions = self.cache_stats["evictions"]
        expirations = self.cache_stats["expirations"]
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0.0

        return {
            "hits": hits,
            "misses": misses,
            "evictions": evictions,
            "expirations": expirations,
            "hit_rate": hit_rate,
        }

    def clear(self):
        self._reset()

    def _sync_ttl_items(self, target_key=None):
        is_target_key_expired = False
        while self.ttl_entries and self.ttl_entries[0][0] <= self.clock():
            entry_tuple = heapq.heappop(self.ttl_entries)
            entry = entry_tuple[1]
            freq = self.key_freq[entry.key]
            del self.cache[entry.key]
            del self.key_freq[entry.key]
            del self.freq_count[freq][entry.key]
            self.cache_stats["expirations"] += 1
            if target_key == entry.key:
                is_target_key_expired = True
            if freq == self.min_freq and len(self.freq_count[freq]) == 0:
                self.min_freq = 1

        return is_target_key_expired
