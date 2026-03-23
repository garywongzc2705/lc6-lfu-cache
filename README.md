# Anthropic — Staff Engineer · Incremental Coding Round #7

## Rules
- Work in `solution.py` only — do not modify test files
- Run tests with: `python -m pytest test_level1.py -v`
- Complete each level before requesting the next
- Add a comment block at the top of each level explaining your approach
- Time yourself — aim for 15–20 minutes per level
- Refactoring between levels is allowed and expected

## Setup
```bash
pip install pytest
```

## Levels
- **Level 1** — available now in `test_level1.py`
- **Level 2** — revealed after Level 1 passes
- **Level 3** — revealed after Level 2 passes
- **Level 4** — revealed after Level 3 passes

## The real test
LFU is asked almost as often as LRU. The naive solution uses a heap
and is O(n log n) for updates. The correct solution is O(1) for both
get and put. Know the three data structures before you start.
The O(1) insight: you don't need to sort all frequencies —
you only ever need to find the minimum.
