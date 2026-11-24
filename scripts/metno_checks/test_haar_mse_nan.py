"""Missing (NaN) target values in the Haar MSE (global_haar_wavelet_reshape_varweighted) (CPU).
Usage: python test_haar_mse_nan.py <repo> [commit_before]
  1) without NaN: identical to the code before the change (level_norm off and on, with and
     without a hole in the domain)
  2) with NaN: the code before the change gives NaN; now the loss is finite, gradients are
     finite, the gradient at the NaN points is 0, and a few NaN change the loss only a little
  3) with NaN in one channel only: the other channels are unchanged
  4) a NaN point counts like a point whose target and prediction are both 0 for the error,
     but it is left out of the field variance (so the result differs from simply writing 0)"""
import subprocess
import sys
import types

import numpy as np
import torch

import weathergen.train.loss_modules.loss_functions as lf

repo = sys.argv[1]
commit = sys.argv[2] if len(sys.argv) > 2 else "HEAD"
src = subprocess.run(["git", "-C", repo, "show", f"{commit}:src/weathergen/train/loss_modules/loss_functions.py"],
                     capture_output=True, text=True, check=True).stdout
old = types.ModuleType("old_lf")
old.__file__ = lf.__file__
exec(compile(src, "old_loss_functions.py", "exec"), old.__dict__)

torch.manual_seed(0)
ny, nx, C, E, L = 90, 70, 3, 2, 4
lat = np.linspace(55.0, 70.0, ny)[:, None].repeat(nx, 1)
lon = np.linspace(0.0, 30.0, nx)[None, :].repeat(ny, 0)
tmpl = "/tmp/tmpl_mse_nan.npz"
np.savez(tmpl, lat=lat.ravel(), lon=lon.ravel(), ny=ny, nx=nx)
coords = torch.tensor(np.stack([lat.ravel(), lon.ravel()], 1), dtype=torch.float32)
g = torch.Generator().manual_seed(1)
Y = torch.randn(ny, nx, C, generator=g).cumsum(0).cumsum(1).reshape(-1, C) * 0.02
X = Y.unsqueeze(0) + 0.3 * torch.randn(E, ny * nx, C, generator=g)
keep = torch.from_numpy(~((lat.ravel() > 66) & (lon.ravel() > 22)))
KW = dict(template_path=tmpl, num_levels=L, var_weight_epsilon=0.1)


def new(y, x, c, **kw):
    return lf.global_haar_wavelet_reshape_varweighted(y, x, c, None, None, **{**KW, **kw})


def prev(y, x, c, **kw):
    return old.global_haar_wavelet_reshape_varweighted(y, x, c, None, None, **{**KW, **kw})


# 1) without NaN: identical ---------------------------------------------------------------
with torch.no_grad():
    for norm in (False, True):
        for y, x, c in ((Y, X, coords), (Y[keep], X[:, keep], coords[keep])):
            a = new(y, x, c, level_norm=norm)[1]
            b = prev(y, x, c, level_norm=norm)[1]
            assert torch.equal(a, b), (norm, a, b)
print("1) without NaN: identical to the code before the change (level_norm off and on)")

# 2) with NaN -----------------------------------------------------------------------------
Yn = Y.clone()
idx = torch.randperm(ny * nx, generator=g)[:60]        # 60 of 6300 points (~1%)
Yn[idx, :] = float("nan")
for norm in (False, True):
    with torch.no_grad():
        before = prev(Yn, X, coords, level_norm=norm)[0]
        ref = new(Y, X, coords, level_norm=norm)[0]
    assert torch.isnan(before), before
    Xg = X.clone().requires_grad_(True)
    loss = new(Yn, Xg, coords, level_norm=norm)[0]
    loss.backward()
    assert torch.isfinite(loss), loss
    assert torch.isfinite(Xg.grad).all()
    assert Xg.grad[:, idx].abs().max() == 0.0
    rel = abs(float(loss.detach()) / float(ref) - 1)
    assert rel < 0.1, rel
    print(f"2) level_norm={norm}: before NaN, now {float(loss.detach()):.6f} (no NaN: {float(ref):.6f}, "
          f"change {100 * rel:.1f}%); gradients finite, 0 at the NaN points")

# 3) NaN in one channel only ---------------------------------------------------------------
Y1 = Y.clone()
Y1[idx, 1] = float("nan")
with torch.no_grad():
    for norm in (False, True):
        a = new(Y1, X, coords, level_norm=norm)[1]
        b = new(Y, X, coords, level_norm=norm)[1]
        assert torch.equal(a[[0, 2]], b[[0, 2]]), (a, b)
        assert torch.isfinite(a).all()
print("3) NaN in one channel: the other channels are unchanged")

# 4) NaN point = target and prediction 0 for the error, but not in the field variance ------
Yz = Y.clone(); Yz[idx, :] = 0.0
Xz = X.clone(); Xz[:, idx, :] = 0.0
with torch.no_grad():
    a = new(Yn, X, coords)[1]
    z = new(Yz, Xz, coords)[1]
    # same weighted errors, different variance: ratio = V(zeros included) / V(NaN excluded)
    Yg = Y.reshape(ny, nx, C)
    m = torch.ones(ny * nx, dtype=torch.bool); m[idx] = False
    v_nan = Y[m].var(0)
    v_zero = Yz.var(0)
    assert torch.allclose(a / z, v_zero / v_nan, rtol=1e-4), (a / z, v_zero / v_nan)
print("4) a NaN point counts as error 0 and is left out of the field variance")
print("ALL HAAR MSE NAN TESTS PASSED")
