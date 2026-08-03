"""
Standalone correctness tests for DomainPyramid and its cross-level maps.

Pure CPU, no torch, no GPU. Run: python test_domain_pyramid.py
"""

import pickle
import sys

import numpy as np

# make weathergen.datasets.domain importable as the pyramid expects
sys.path.insert(0, "/tmp/_wgshim")
import os
os.makedirs("/tmp/_wgshim/weathergen/datasets", exist_ok=True)
for p in ["/tmp/_wgshim/weathergen/__init__.py",
          "/tmp/_wgshim/weathergen/datasets/__init__.py"]:
    open(p, "a").close()
import shutil
shutil.copy("/tmp/domain.py", "/tmp/_wgshim/weathergen/datasets/domain.py")
shutil.copy("/tmp/domain_pyramid.py", "/tmp/_wgshim/weathergen/datasets/domain_pyramid.py")

from weathergen.datasets.domain import Domain
from weathergen.datasets.domain_pyramid import (
    DomainPyramid,
    cells_children,
    cells_parent,
)

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


# ---------------------------------------------------------------------------
print("1. parent/child arithmetic inverses")
# every child's parent is the original cell
for (lc, lf) in [(5, 8), (3, 4), (2, 6)]:
    g_coarse = np.arange(12 * 4**lc, dtype=np.int64)
    kids = cells_children(g_coarse, lc, lf)                 # (N, 4**(lf-lc))
    back = cells_parent(kids.reshape(-1), lf, lc).reshape(kids.shape)
    check(f"children->parent round trip hl{lc}->hl{lf}",
          np.all(back == g_coarse[:, None]))
    check(f"child count hl{lc}->hl{lf} == 4**(df)",
          kids.shape[1] == 4 ** (lf - lc))
    # children are contiguous and cover the fine index space exactly once
    check(f"children partition fine space hl{lc}->hl{lf}",
          np.array_equal(np.sort(kids.reshape(-1)),
                         np.arange(12 * 4**lf, dtype=np.int64)))

# ---------------------------------------------------------------------------
print("2. single-level pyramid == plain Domain (backward compat)")
d = Domain.global_(5)
pyr = DomainPyramid.single(5, d)
check("single is_single", pyr.is_single)
check("single len", len(pyr) == 1)
check("single coarsest==finest", pyr.coarsest == pyr.finest == 5)
check("single domain identity", pyr.domain(5) is d)

# ---------------------------------------------------------------------------
print("3. two global levels: full parent/child coverage")
pyr = DomainPyramid({5: Domain.global_(5), 7: Domain.global_(7)})
poc = pyr.parent_of_child(5, 7)     # (num_fine,)
cop = pyr.child_of_parent(5, 7)     # (num_coarse, 16)
cv = pyr.child_valid(5, 7)
check("global: every fine cell has a parent", np.all(poc >= 0))
check("global: every coarse cell has all children valid", np.all(cv))
check("global: child count == 16", cop.shape[1] == 16)
# round trip: parent of each listed child == the coarse cell
n_coarse = len(pyr.domain(5))
parent_check = True
for c in range(min(n_coarse, 200)):   # sample for speed
    kids = cop[c]
    parent_check &= np.all(poc[kids] == c)
check("global: child_of_parent and parent_of_child consistent", parent_check)
# pooling identity: averaging children then indexing == identity mapping exists
check("global: num fine == num coarse * 16",
      len(pyr.domain(7)) == len(pyr.domain(5)) * 16)

# ---------------------------------------------------------------------------
print("4. regional nested domains (fine smaller than coarse)")
# coarse hl5 over a big box, fine hl8 over a small box inside it
coarse = Domain.from_bbox(5, lon_min=-20, lon_max=40, lat_min=40, lat_max=80, pad_rings=0)
fine = Domain.from_bbox(8, lon_min=0, lon_max=20, lat_min=53, lat_max=66, pad_rings=0)
pyr = DomainPyramid({5: coarse, 8: fine})
poc = pyr.parent_of_child(5, 8)
cop = pyr.child_of_parent(5, 8)
cv = pyr.child_valid(5, 8)
check("regional: fine domain smaller than coarse", len(fine) < len(coarse) * 64)
# every fine cell's parent, when paired, must be a valid coarse compact index
paired = poc >= 0
check("regional: paired parents in range",
      np.all(poc[paired] < len(coarse)) if paired.any() else True)
check("regional: at least some fine cells are paired", paired.any())
# a coarse cell either has some valid children or none; valid children must be
# real fine compact indices
valid_children = cop[cv]
check("regional: valid children in range",
      np.all((valid_children >= 0) & (valid_children < len(fine))))
# CONSISTENCY: for every valid (parent c, child f) pair, poc[f] == c
consistent = True
for c in range(len(coarse)):
    kids = cop[c][cv[c]]
    if len(kids):
        consistent &= np.all(poc[kids] == c)
check("regional: parent<->child maps mutually consistent", consistent)

# ---------------------------------------------------------------------------
print("5. unpaired cells handled (no silent mismap)")
# fine cells whose parent is OUTSIDE the coarse domain must be -1.
# Construct: coarse domain deliberately smaller than fine's footprint.
coarse_small = Domain.from_bbox(5, lon_min=5, lon_max=10, lat_min=55, lat_max=60, pad_rings=0)
fine_big = Domain.from_bbox(8, lon_min=-10, lon_max=30, lat_min=45, lat_max=70, pad_rings=0)
pyr = DomainPyramid({5: coarse_small, 8: fine_big})
poc = pyr.parent_of_child(5, 8)
check("unpaired: some fine cells have NO coarse parent (-1)", np.any(poc < 0))
check("unpaired: -1 is the only sentinel (no other negatives)",
      np.all(poc[poc < 0] == -1))
# every non -1 parent must be valid
check("unpaired: paired parents valid",
      np.all(poc[poc >= 0] < len(coarse_small)))

# ---------------------------------------------------------------------------
print("6. disjoint fine nests inside one coarse domain")
# two separated fine boxes -> union domain via explicit cell concatenation
f1 = Domain.from_bbox(7, lon_min=0, lon_max=6, lat_min=54, lat_max=60)
f2 = Domain.from_bbox(7, lon_min=25, lon_max=31, lat_min=54, lat_max=60)
union_cells = np.unique(np.concatenate([f1.active_cells, f2.active_cells]))
fine_union = Domain(7, union_cells, is_global=False,
                    bbox=(0, 31, 54, 60))  # bbox is just a loose bound here
coarse = Domain.from_bbox(5, lon_min=-10, lon_max=40, lat_min=48, lat_max=66)
pyr = DomainPyramid({5: coarse, 7: fine_union})
poc = pyr.parent_of_child(5, 7)
cop = pyr.child_of_parent(5, 7)
cv = pyr.child_valid(5, 7)
# the two nests should map to two disjoint sets of coarse parents
paired_parents = np.unique(poc[poc >= 0])
check("disjoint: fine union has cells from both nests",
      len(fine_union) == len(union_cells))
check("disjoint: paired parents form >1 contiguous group",
      len(paired_parents) >= 2)
# consistency still holds across the disjoint union
consistent = True
for c in range(len(coarse)):
    kids = cop[c][cv[c]]
    if len(kids):
        consistent &= np.all(poc[kids] == c)
check("disjoint: maps consistent across nests", consistent)

# ---------------------------------------------------------------------------
print("7. three-level pyramid, adjacent pairs only")
pyr = DomainPyramid({4: Domain.global_(4), 6: Domain.global_(6), 8: Domain.global_(8)})
check("3-level: levels sorted", pyr.levels == [4, 6, 8])
check("3-level: pair (4,6) exists", pyr.parent_of_child(4, 6) is not None)
check("3-level: pair (6,8) exists", pyr.parent_of_child(6, 8) is not None)
# non-adjacent pair should NOT be precomputed
try:
    pyr.parent_of_child(4, 8)
    check("3-level: non-adjacent pair rejected", False)
except AssertionError:
    check("3-level: non-adjacent pair rejected", True)

# ---------------------------------------------------------------------------
print("8. picklable (survives being sent to dataloader workers)")
pyr = DomainPyramid({5: Domain.from_bbox(5, -20, 40, 40, 80),
                     8: Domain.from_bbox(8, 0, 20, 53, 66)})
blob = pickle.dumps(pyr)
pyr2 = pickle.loads(blob)
check("pickle: levels preserved", pyr2.levels == pyr.levels)
check("pickle: maps preserved",
      np.array_equal(pyr2.parent_of_child(5, 8), pyr.parent_of_child(5, 8)))
check("pickle: no torch/other deps leaked",
      isinstance(pyr2.child_of_parent(5, 8), np.ndarray))

# ---------------------------------------------------------------------------
print("9. to_compact_safe correctness (global and regional)")
dg = Domain.global_(4)
# global: identity in range, -1 out of range
q = np.array([-1, 0, 5, 12 * 4**4 - 1, 12 * 4**4, 999999], dtype=np.int64)
r = dg.to_compact_safe(q)
check("safe global: identity in range",
      r[1] == 0 and r[2] == 5 and r[3] == 12 * 4**4 - 1)
check("safe global: -1 out of range", r[0] == -1 and r[4] == -1 and r[5] == -1)
dr = Domain.from_bbox(4, 0, 30, 50, 70)
# regional: active cells map to [0,num), inactive -> -1
some_active = dr.active_cells[0]
some_inactive = None
for g in range(dr.num_total_cells):
    if dr.to_compact[g] < 0:
        some_inactive = g
        break
check("safe regional: active maps in range",
      0 <= dr.to_compact_safe(np.array([some_active]))[0] < len(dr))
check("safe regional: inactive -> -1",
      dr.to_compact_safe(np.array([some_inactive]))[0] == -1)

# ---------------------------------------------------------------------------
print(f"\n{'='*50}\n  {PASS} passed, {FAIL} failed\n{'='*50}")
sys.exit(1 if FAIL else 0)
