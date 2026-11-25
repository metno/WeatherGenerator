"""Random source masking must not empty a stream on one rank (CPU).

p01 (2 nodes) deadlocked at the first step: with random cell masking of the inputs, a
spoofed window (2 points in 2 cells, e.g. MEPS before 2020) or a sparse stream can lose
all its source cells on one rank; that rank skips the stream's embedding, the FSDP
collectives get out of step -> NCCL watchdog timeout. keep_one_source_cell() keeps one
cell with tokens in that case.

Checks:
  1. mask that drops every cell with tokens -> exactly one data cell kept, rest unchanged
  2. mask that already keeps a data cell     -> unchanged (same object)
  3. all-false mask (diagnostic stream)      -> unchanged
  4. window without tokens                   -> unchanged
  5. torch and numpy masks; input mask not modified
  6. through TokenizerMasking.get_source's mask path: a spoof-like window (2 cells) under
     a 75% random mask never ends with zero source tokens (old code: ~6% of the samples)
"""
import numpy as np
import torch

from weathergen.datasets.tokenizer_masking import keep_one_source_cell

rng = np.random.default_rng(0)
ncells = 200
lens = [[] for _ in range(ncells)]
data_cells = [17, 123]                       # spoof-like: 2 points in 2 cells
for c in data_cells:
    lens[c] = [1]

# 1. mask keeps 75% of the cells but neither data cell
m = np.ones(ncells, dtype=bool)
m[:50] = False
m[data_cells] = False
for mask in (m.copy(), torch.from_numpy(m.copy())):
    before = np.array(mask).copy()
    new = keep_one_source_cell(mask, lens, rng, "TEST")
    new_np = np.asarray(new)
    assert np.array_equal(np.asarray(mask), before), "input mask modified"
    assert new_np[data_cells].sum() == 1, new_np[data_cells]
    diff = np.flatnonzero(new_np != before)
    assert len(diff) == 1 and diff[0] in data_cells, diff
    assert type(new) is type(mask)
print("1. all data cells masked -> one kept: ok (numpy and torch)")

# 2. already keeps a data cell
m2 = m.copy()
m2[data_cells[0]] = True
assert keep_one_source_cell(m2, lens, rng) is m2
print("2. data cell already kept -> unchanged: ok")

# 3. diagnostic stream: nothing kept on purpose
m3 = np.zeros(ncells, dtype=bool)
assert keep_one_source_cell(m3, lens, rng) is m3
print("3. all-false mask -> unchanged: ok")

# 4. no tokens at all
assert keep_one_source_cell(m, [[] for _ in range(ncells)], rng) is m
print("4. no tokens -> unchanged: ok")

# 6. frequency with the old and the new code, random 75% masks (rate jitter as in Masker)
n, empty_old, empty_new = 20000, 0, 0
for _ in range(n):
    keep = np.abs(rng.normal(0.75, 1.0 / (2.5 * np.pi)))
    mk = rng.uniform(0, 1, ncells) < keep
    has = np.array([len(x) > 0 for x in lens])
    empty_old += not (mk & has).any()
    mk2 = np.asarray(keep_one_source_cell(mk, lens, rng))
    empty_new += not (mk2 & has).any()
print(f"6. spoof-like window, 75% masks: empty source in {100 * empty_old / n:.1f}% "
      f"(old) -> {100 * empty_new / n:.1f}% (new)")
assert empty_new == 0 and empty_old > 0
print("OK")
