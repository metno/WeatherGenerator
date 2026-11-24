"""haar_random_shift in the three Haar losses (CPU).
Usage: python test_haar_random_shift.py <repo> [commit_before]
  1) option off, or no gradients (validation): identical to the code before the change
  2) option on in training: finite, gradients, changes from call to call
  3) the point of it: with the fixed lattice, steps ON the block edges cost a different
     amount than the same steps shifted by a few points; with the random offset the
     average cost no longer depends on where the steps are."""
import subprocess
import sys
import types

import numpy as np
import torch

import weathergen.train.loss_modules.loss_functions as lf

repo = sys.argv[1]
commit = sys.argv[2] if len(sys.argv) > 2 else "4dfa37a"
src = subprocess.run(["git", "-C", repo, "show", f"{commit}:src/weathergen/train/loss_modules/loss_functions.py"],
                     capture_output=True, text=True, check=True).stdout
old = types.ModuleType("old_lf")
old.__file__ = lf.__file__
exec(compile(src, "old_loss_functions.py", "exec"), old.__dict__)

ny, nx = 160, 128
lat = np.linspace(55.0, 70.0, ny)[:, None].repeat(nx, 1)
lon = np.linspace(0.0, 30.0, nx)[None, :].repeat(ny, 0)
np.savez("/tmp/tmpl_shift.npz", lat=lat.ravel(), lon=lon.ravel(), ny=ny, nx=nx)  # flat, as the CARRA2 template
coords = torch.tensor(np.stack([lat.ravel(), lon.ravel()], 1), dtype=torch.float32)
g = torch.Generator().manual_seed(0)
C, E = 3, 2
# smooth target (no fine structure)
yy, xx = torch.meshgrid(torch.linspace(0, 1, ny), torch.linspace(0, 1, nx), indexing="ij")
base = torch.stack([torch.sin(3 * yy + k) * torch.cos(2 * xx - k) for k in range(C)], -1)  # (ny,nx,C)
tgt = base.reshape(-1, C)


def blocky(offset, block=32, amp=0.3, seed=1):
    """member perturbation: random constant value per block, blocks start at `offset`."""
    gg = torch.Generator().manual_seed(seed)
    vals = amp * torch.randn(E, ny // block + 2, nx // block + 2, C, generator=gg)
    iy = (torch.arange(ny) + (block - offset)) // block
    ix = (torch.arange(nx) + (block - offset)) // block
    return vals[:, iy][:, :, ix].reshape(E, -1, C)


fns = {
    "crps": dict(new=lf.global_haar_wavelet_reshape_varweighted_crps,
                 old=old.global_haar_wavelet_reshape_varweighted_crps,
                 kw=dict(num_levels=5, member_ll_mse_weight=0.05, crps_alpha=0.95)),
    "mse": dict(new=lf.global_haar_wavelet_reshape_varweighted,
                old=old.global_haar_wavelet_reshape_varweighted,
                kw=dict(num_levels=4, var_weight_epsilon=0.1)),
    "energy": dict(new=lf.global_haar_energy_reshape, old=old.global_haar_energy_reshape,
                   kw=dict(num_levels=4, ensemble_mode="per_member")),
}
pred = (tgt.unsqueeze(0) + blocky(0)).requires_grad_(True)

# 1) off / no-grad identical to the old code
for name, f in fns.items():
    a = f["old"](tgt, pred, coords, None, None, template_path="/tmp/tmpl_shift.npz", **f["kw"])[0]
    b = f["new"](tgt, pred, coords, None, None, template_path="/tmp/tmpl_shift.npz", **f["kw"])[0]
    with torch.no_grad():
        c = f["new"](tgt, pred, coords, None, None, template_path="/tmp/tmpl_shift.npz",
                     haar_random_shift=True, **f["kw"])[0]
    assert torch.equal(a, b) and torch.equal(a.detach(), c), (name, a, b, c)
print("1) option off, and validation (no gradients): identical to the old code")

# 2) on in training: finite, gradient, varies between calls
for name, f in fns.items():
    vals = []
    for _ in range(5):
        l = f["new"](tgt, pred, coords, None, None, template_path="/tmp/tmpl_shift.npz",
                     haar_random_shift=True, **f["kw"])[0]
        vals.append(l.item())
    l.backward()
    assert all(np.isfinite(vals)) and len(set(vals)) > 1 and pred.grad.abs().sum() > 0, (name, vals)
    pred.grad = None
print("2) option on in training: finite, gradients flow, offset changes between calls")

# 3) lattice preference: steps on the block edges (offset 0) vs the same steps shifted by 11
print("3) mean loss, blocky members with steps ON the 32-point lattice vs shifted by 11 points:")
for name, f in fns.items():
    res = {}
    for shift_on in (False, True):
        for off in (0, 11):
            p = tgt.unsqueeze(0) + blocky(off)
            n = 200 if shift_on else 1
            torch.manual_seed(0)
            import random as _r
            _r.seed(0)
            vals = [f["new"](tgt, p.requires_grad_(True), coords, None, None,
                             template_path="/tmp/tmpl_shift.npz", haar_random_shift=shift_on,
                             **f["kw"])[0].item() for _ in range(n)]
            res[(shift_on, off)] = float(np.mean(vals))
    fixed = res[(False, 11)] / res[(False, 0)]
    rnd = res[(True, 11)] / res[(True, 0)]
    print(f"   {name:6s}: fixed lattice  aligned {res[(False, 0)]:.4f}  shifted {res[(False, 11)]:.4f}"
          f"  (ratio {fixed:.2f}) | random offset  aligned {res[(True, 0)]:.4f}"
          f"  shifted {res[(True, 11)]:.4f}  (ratio {rnd:.2f})")
    # the shifted pattern differs slightly at the domain edges, so exact equality is not
    # expected; the lattice preference must shrink by a large factor
    assert abs(np.log(rnd)) < 0.25 * abs(np.log(fixed)) or abs(np.log(rnd)) < 0.1, (name, fixed, rnd)
print("ALL HAAR RANDOM SHIFT TESTS PASSED")
