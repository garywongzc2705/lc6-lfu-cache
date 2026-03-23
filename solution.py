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
    def __init__(self, key, val, expires_at=None):
        self.key = key
        self.val = val
        self.expires_at = expires_at


class LFUCache:
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.clock = time.time
        self.on_evicts = []
        self.on_expires = []
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
        is_new = True
        if key not in self.cache and len(self.cache) == self.capacity:
            self._evict()

        if key in self.cache:
            is_new = False
            self._update_ttl(self.cache[key])

        entry = (
            self.cache[key]
            if key in self.cache
            else CacheEntry(key, value, self.clock() + ttl if ttl else None)
        )
        self._update_freqency(entry)
        entry.val = value
        self.cache[key] = entry

        if ttl:
            heapq.heappush(self.ttl_entries, (self.clock() + ttl, entry))

        return is_new

    def size(self):
        self._sync_ttl_items()
        return len(self.cache)

    def contains(self, key):
        self._sync_ttl_items()
        return key in self.cache

    def _update_ttl(self, target: CacheEntry):
        original_size = len(self.ttl_entries)
        self.ttl_entries = [
            entry for entry in self.ttl_entries if entry[1].key != target.key
        ]
        if original_size > len(self.ttl_entries):
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
        cache_entry = self.cache[key]

        del self.cache[key]
        del self.key_freq[key]
        self.cache_stats["evictions"] += 1
        for evict_cb in self.on_evicts:
            evict_cb(key, cache_entry.val)

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

            for expire_cb in self.on_expires:
                expire_cb(entry.key, entry.val)

        return is_target_key_expired

    def on_evict(self, fn):
        self.on_evicts.append(fn)

    def on_expire(self, fn):
        self.on_expires.append(fn)

    def remove(self, key):
        self._sync_ttl_items()
        if not key in self.cache:
            return -1

        entry = self.cache[key]
        freq = self.key_freq[entry.key]
        del self.cache[entry.key]
        del self.key_freq[entry.key]
        del self.freq_count[freq][entry.key]
        self._update_ttl(entry)
        if freq == self.min_freq and len(self.freq_count[freq]) == 0:
            self.min_freq = 1
        return entry.val

    def frequency(self, key):
        self._sync_ttl_items()
        if key not in self.cache:
            return 0
        return self.key_freq[key]

    def most_frequent(self):
        freq_counts = self._get_freq_map_entries()
        result = []
        target_size = 5
        for i in range(len(freq_counts) - 1, -1, -1):
            freq_keys_tuple = freq_counts[i]
            freq, keys = freq_keys_tuple[0], freq_keys_tuple[1]

            take = min(target_size - len(result), len(keys))
            result.extend([(key, freq) for key in reversed(list(keys.keys())[-take:])])
            if len(result) == target_size:
                break

        return result

    def least_frequent(self):
        freq_counts = self._get_freq_map_entries()
        result = []

        target_size = 1
        for i in range(len(freq_counts)):
            freq_keys_tuple = freq_counts[i]
            freq, keys = freq_keys_tuple[0], freq_keys_tuple[1]

            take = min(target_size - len(result), len(keys))
            result.extend([(key, freq) for key in (list(keys.keys())[:take])])
            if len(result) == target_size:
                break

        return result[0] if len(result) == 1 else None

    def _get_freq_map_entries(self) -> list[tuple]:
        self._sync_ttl_items()
        freq_counts = []
        for freq, keys in self.freq_count.items():
            freq_counts.append((freq, keys))

        freq_counts = sorted(
            [(freq, keys) for freq, keys in self.freq_count.items() if keys]
        )
        return freq_counts

    def warm(self, entries):
        pre_evictions = self.cache_stats["evictions"]
        inserted, skipped = 0, 0

        for entry in entries:
            key = entry[0]
            value = entry[1]
            ttl = entry[2] if len(entry) > 2 else None

            if key in self.cache:
                skipped += 1
                continue

            self.put(key, value, ttl)
            inserted += 1

        evicted = self.cache_stats["evictions"] - pre_evictions
        return {"inserted": inserted, "skipped": skipped, "evicted": evicted}

    def snapshot(self):
        self._sync_ttl_items()
        cache_stats = self.stats()
        freq_counts = self._get_freq_map_entries()
        entries = []
        for i in range(len(freq_counts) - 1, -1, -1):
            freq_keys_tuple = freq_counts[i]

            freq, keys = freq_keys_tuple[0], freq_keys_tuple[1]
            for key in keys:
                entry = self.cache[key]
                entries.append(
                    {
                        "key": entry.key,
                        "value": entry.val,
                        "frequency": freq,
                        "expires_at": entry.expires_at,
                    }
                )

        return {"entries": entries, "capacity": self.capacity, "stats": cache_stats}

    def restore(self, snapshot):
        if (
            "entries" not in snapshot
            or "capacity" not in snapshot
            or "stats" not in snapshot
        ):
            raise ValueError("invalid snapshot format")

        self._reset()
        self.capacity = snapshot["capacity"]
        self.cache_stats = {**snapshot["stats"]}
        self.cache_stats.pop("hit_rate", None)  # hit_rate is derived, not stored

        now = self.clock()
        min_freq = float("inf")

        for e in snapshot["entries"]:
            expires_at = e.get("expires_at")
            if expires_at is not None and expires_at <= now:
                continue  # skip already expired

            key = e["key"]
            freq = e["frequency"]
            entry = CacheEntry(key, e["value"], expires_at)
            self.cache[key] = entry
            self.key_freq[key] = freq
            self.freq_count[freq][key] = key
            min_freq = min(min_freq, freq)

            if expires_at is not None:
                heapq.heappush(self.ttl_entries, (expires_at, entry))

        self.min_freq = min_freq if min_freq != float("inf") else 1
