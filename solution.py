from collections import OrderedDict, defaultdict

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
    def __init__(self, key, val):
        self.key = key
        self.val = val


class LFUCache:
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.cache = {}
        self.key_freq = defaultdict(int)
        self.freq_count = defaultdict(OrderedDict)
        self.min_freq = 1

    def get(self, key):
        if key not in self.cache:
            return -1
        entry = self.cache[key]
        self._update_freqency(entry)

        return entry.val

    def put(self, key, value):
        if key not in self.cache and len(self.cache) == self.capacity:
            self._evict()
            self.min_freq = 1

        entry = self.cache[key] if key in self.cache else CacheEntry(key, value)
        self._update_freqency(entry)
        entry.val = value
        self.cache[key] = entry

    def size(self):
        return len(self.cache)

    def contains(self, key):
        return key in self.cache

    def _update_freqency(self, entry: CacheEntry):
        old_freq = self.key_freq[entry.key]
        new_freq = old_freq + 1
        self.freq_count[new_freq][entry.key] = entry.key
        self.key_freq[entry.key] = new_freq

        if entry.key in self.freq_count[old_freq]:
            del self.freq_count[old_freq][entry.key]

        if old_freq == self.min_freq and len(self.freq_count[old_freq]) == 0:
            self.min_freq = old_freq + 1

    def _evict(self):
        entry_tuple = self.freq_count[self.min_freq].popitem(last=False)
        key = entry_tuple[0]
        del self.cache[key]
        del self.key_freq[key]
