"""level_norm in the Haar CRPS and Haar MSE losses (CPU).
Usage: python test_haar_level_norm.py <repo> [commit_before]
  1) option off: identical to the code before the change (both losses)
  2) option on: equal to an independent implementation of the formulas, on a field with
     invalid cells and padding; a hole in the domain does not blow up the loss
  3) option on: the loss does not change when all values are multiplied by a constant
     (dimensionless); with the option off it does
  4) option on: on a field whose detail grows ~4x per level, every level contributes a
     similar amount (with the option off the coarsest level dominates)
  5) option on, in training with haar_random_shift: finite, gradients flow"""
import math
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
ny, nx, C, E, L = 90, 70, 3, 2, 4       # not multiples of 16: padding is exercised
lat = np.linspace(55.0, 70.0, ny)[:, None].repeat(nx, 1)
lon = np.linspace(0.0, 30.0, nx)[None, :].repeat(ny, 0)
tmpl = "/tmp/tmpl_level_norm.npz"
np.savez(tmpl, lat=lat.ravel(), lon=lon.ravel(), ny=ny, nx=nx)
coords_full = np.stack([lat.ravel(), lon.ravel()], 1)


def red_field(seed, growth=4.0):
    """sum of Haar-like detail with RMS growing by `growth` per level (coarse = large)."""
    g = torch.Generator().manual_seed(seed)
    f = torch.zeros(ny, nx)
    for l in range(L + 1):
        b = 2 ** l
        n = torch.randn(ny // b + 2, nx // b + 2, generator=g)
        up = n.repeat_interleave(b, 0).repeat_interleave(b, 1)[:ny, :nx]
        f = f + (growth ** l) * up * 0.01
    return f


Y = torch.stack([red_field(10 + c) for c in range(C)], -1).reshape(-1, C)
X = torch.stack([torch.stack([red_field(100 + 10 * m + c) * 0.3 for c in range(C)], -1).reshape(-1, C)
                 for m in range(E)]) + Y.unsqueeze(0)
coords = torch.tensor(coords_full, dtype=torch.float32)

# a domain with a hole: drop a block of points (cells there become invalid)
keep = ~((lat.ravel() > 66) & (lon.ravel() > 22))
Yh, Xh, ch = Y[keep], X[:, keep], coords[torch.from_numpy(keep)]

CRPS = dict(template_path=tmpl, num_levels=L, var_weight_epsilon=0.1, crps_alpha=0.95,
            member_ll_mse_weight=0.05, level_weights=[1.0] * L)
MSE = dict(template_path=tmpl, num_levels=L, var_weight_epsilon=0.1)


def crps_new(y, x, c, **kw):
    return lf.global_haar_wavelet_reshape_varweighted_crps(y, x, c, None, None, **{**CRPS, **kw})[1]


def mse_new(y, x, c, **kw):
    return lf.global_haar_wavelet_reshape_varweighted(y, x, c, None, None, **{**MSE, **kw})[1]


# 1) off: identical to the old code -------------------------------------------------
with torch.no_grad():
    for y, x, c in ((Y, X, coords), (Yh, Xh, ch)):
        a = crps_new(y, x, c)
        b = old.global_haar_wavelet_reshape_varweighted_crps(y, x, c, None, None, **CRPS)[1]
        assert torch.equal(a, b), (a, b)
        a = mse_new(y, x, c)
        b = old.global_haar_wavelet_reshape_varweighted(y, x, c, None, None, **MSE)[1]
        assert torch.equal(a, b), (a, b)
print("1) option off: identical to the code before the change (CRPS and MSE)")


# 2) on: independent implementation -------------------------------------------------
def haar(F):
    a, b, c, d = F[..., 0::2, 0::2], F[..., 0::2, 1::2], F[..., 1::2, 0::2], F[..., 1::2, 1::2]
    return (a + b + c + d) / 2, ((a + c) - (b + d)) / 2, ((a + b) - (c + d)) / 2, ((a + d) - (b + c)) / 2


def reference(y, x, c, eps=0.1, alpha=0.95, mu=0.05):
    """formulas written out, double precision; regrid and valid mask from the shared helper."""
    tm, (gy, gx) = lf._load_haar_template(reference, tmpl, "", "ref")
    P = y.shape[0]
    t_flat, p_flat, valid_cell, _ = lf._reshape_regrid_to_grid(
        y.repeat(1, E), x.permute(1, 0, 2).reshape(P, E * C), c, tm, gy, gx, E * C,
        torch.device("cpu"))
    Yg = t_flat[:, :C].reshape(gy, gx, C).permute(2, 0, 1).double()
    Xg = p_flat.reshape(gy, gx, E, C).permute(2, 3, 0, 1).double()
    v = valid_cell.view(gy, gx).double()
    f = 2**L
    Ny, Nx = math.ceil(gy / f) * f, math.ceil(gx / f) * f
    Yp = torch.zeros(C, Ny, Nx, dtype=torch.double); Yp[:, :gy, :gx] = Yg
    Xp = torch.zeros(E, C, Ny, Nx, dtype=torch.double); Xp[..., :gy, :gx] = Xg
    Vp = torch.zeros(Ny, Nx, dtype=torch.double); Vp[:gy, :gx] = v
    vm = valid_cell.view(gy, gx)
    V = torch.stack([Yg[k][vm].var() for k in range(C)])
    gE = alpha / (E * (E - 1)) + (1 - alpha) / E**2
    crps_tot = torch.zeros(C, dtype=torch.double)
    mse_tot = torch.zeros(C, dtype=torch.double)
    Yl, Xl, Vl = Yp, Xp, Vp
    for _ in range(L):
        YLL, *Yb = haar(Yl)
        XLL, *Xb = haar(Xl)
        blocks = [Vl[0::2, 0::2], Vl[0::2, 1::2], Vl[1::2, 0::2], Vl[1::2, 1::2]]
        v_all = torch.stack(blocks).min(0).values
        en = Yb[0] ** 2 + Yb[1] ** 2 + Yb[2] ** 2                          # (C,h,w)
        s2 = (en * v_all).sum((1, 2)) / (3 * v_all.sum())                   # (C,)
        w = v_all / (en + eps * s2.view(C, 1, 1))
        for yb, xb in zip(Yb, Xb):
            cr = (xb - yb).abs().mean(0) - gE * (xb[0] - xb[1]).abs()
            crps_tot += (w * cr).sum((1, 2)) / w.sum((1, 2)) / (3 * s2.sqrt())
            mse_tot += (w * (xb.mean(0) - yb) ** 2).sum((1, 2)) / w.sum((1, 2)) / (3 * s2)
        Yl, Xl, Vl = YLL, XLL, v_all
    ll = mu * ((Xl / f - Yl / f) ** 2).mean((0, 2, 3)) / V
    return crps_tot / L + ll, mse_tot / L


res = {}
with torch.no_grad():
    for y, x, c, name in ((Y, X, coords, "full domain"), (Yh, Xh, ch, "domain with a hole")):
        rc, rm = reference(y, x, c)
        a = crps_new(y, x, c, level_norm=True).double()
        b = mse_new(y, x, c, level_norm=True).double()
        assert torch.allclose(a, rc, rtol=1e-4), (name, a, rc)
        assert torch.allclose(b, rm, rtol=1e-4), (name, b, rm)
        print(f"2) option on, {name}: CRPS {a.numpy()} = formulas {rc.numpy()}")
        print(f"   option on, {name}: MSE  {b.numpy()} = formulas {rm.numpy()}")
        res[name] = (a, b)
    # the hole must not create a fine-scale "error" at its edge (members := target there)
    for i, nm in enumerate(("CRPS", "MSE")):
        r = (res["domain with a hole"][i] / res["full domain"][i]).max()
        assert r < 1.3, (nm, r)
    print("   the hole changes the loss by less than 30% (no edge artefact)")

# 3) dimensionless --------------------------------------------------------------------
with torch.no_grad():
    k = 7.0
    for fn, name in ((crps_new, "CRPS"), (mse_new, "MSE")):
        on1 = fn(Y, X, coords, level_norm=True, member_ll_mse_weight=0.0) if name == "CRPS" else fn(Y, X, coords, level_norm=True)
        onk = fn(Y * k, X * k, coords, level_norm=True, member_ll_mse_weight=0.0) if name == "CRPS" else fn(Y * k, X * k, coords, level_norm=True)
        off1 = fn(Y, X, coords)
        offk = fn(Y * k, X * k, coords)
        assert torch.allclose(on1, onk, rtol=1e-4), (name, on1, onk)
        print(f"3) {name}: values x{k:g} -> loss ratio {float((onk / on1).mean()):.4f} with level_norm, "
              f"{float((offk / off1).mean()):.4f} without")

# 4) balance between levels -----------------------------------------------------------
with torch.no_grad():
    for norm in (False, True):
        contrib = []
        for l in range(L):
            lw = [0.0] * L
            lw[l] = 1.0
            v = crps_new(Y, X, coords, level_weights=lw, member_ll_mse_weight=0.0, level_norm=norm)
            contrib.append(float(v.mean()))
        sh = np.array(contrib) / sum(contrib)
        print(f"4) CRPS share per level L0..L{L - 1}, level_norm={norm}: " + " ".join(f"{v:.3f}" for v in sh))
        if norm:
            assert sh.max() / sh.min() < 3.0, sh
        else:
            assert sh[-1] > 0.5, sh

# 5) training with random offset: finite, gradients --------------------------------------
Xg = X.clone().requires_grad_(True)
l1, _ = lf.global_haar_wavelet_reshape_varweighted_crps(Y, Xg, coords, None, None,
                                                        **CRPS, level_norm=True, haar_random_shift=True)
l2, _ = lf.global_haar_wavelet_reshape_varweighted(Y, Xg, coords, None, None,
                                                   **MSE, level_norm=True, haar_random_shift=True)
(l1 + l2).backward()
assert torch.isfinite(l1) and torch.isfinite(l2) and torch.isfinite(Xg.grad).all()
assert Xg.grad.abs().sum() > 0
print("5) option on in training with the random offset: finite, gradients flow")
print("ALL HAAR LEVEL_NORM TESTS PASSED")
