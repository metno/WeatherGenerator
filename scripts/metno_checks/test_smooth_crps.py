"""global_smooth_crps: CRPS of Gaussian-smoothed fields (CPU).
Usage: python test_smooth_crps.py
  1) equal to an independent implementation (scipy Gaussian filter), with a hole in the
     domain and missing target values
  2) no block lattice: blocky members cost the same wherever their block edges are
  3) dimensionless: multiplying all values by a constant does not change the loss
  4) large-scale anchor: members shifted together away from the target cost more and more;
     a perfect forecast costs 0
  5) one member: absolute error of the smoothed fields
  6) missing values: finite, gradients finite and 0 at the missing points; a hole does
     not blow up the loss (no edge artefact)"""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import weathergen.train.loss_modules.loss_functions as lf

torch.manual_seed(0)
ny, nx, C, E = 96, 80, 3, 2
lat = np.linspace(55.0, 70.0, ny)[:, None].repeat(nx, 1)
lon = np.linspace(0.0, 30.0, nx)[None, :].repeat(ny, 0)
tmpl = "/tmp/tmpl_smooth_crps.npz"
np.savez(tmpl, lat=lat.ravel(), lon=lon.ravel(), ny=ny, nx=nx)
coords = torch.tensor(np.stack([lat.ravel(), lon.ravel()], 1), dtype=torch.float32)
g = torch.Generator().manual_seed(1)
Y = torch.randn(ny, nx, C, generator=g).cumsum(0).cumsum(1).reshape(-1, C) * 0.02
X = Y.unsqueeze(0) + 0.3 * torch.randn(E, ny * nx, C, generator=g)
SIG_KM, GRID_KM, ALPHA = 10.0, 2.5, 0.95          # sigma = 4 grid points
KW = dict(template_path=tmpl, sigma_km=SIG_KM, grid_km=GRID_KM, crps_alpha=ALPHA)


def loss_chs(y, x, c, **kw):
    return lf.global_smooth_crps(y, x, c, None, None, **{**KW, **kw})[1]


# 1) independent implementation -------------------------------------------------------------
def reference(y, x, c, alpha=ALPHA, min_cov=0.5):
    tm, (gy, gx) = lf._load_haar_template(reference, tmpl, "", "ref")
    P = y.shape[0]
    Ee = x.shape[0]
    nan = torch.isnan(y)
    y0 = torch.where(nan, torch.zeros_like(y), y)
    x0 = torch.where(nan.unsqueeze(0), torch.zeros_like(x), x)
    t_flat, p_flat, valid_cell, _ = lf._reshape_regrid_to_grid(
        torch.cat([y0.repeat(1, Ee), nan.float()], 1),
        torch.cat([x0.permute(1, 0, 2).reshape(P, Ee * C), nan.float()], 1),
        c, tm, gy, gx, Ee * C + C, torch.device("cpu"))
    Yg = t_flat[:, :C].reshape(gy, gx, C).permute(2, 0, 1).double().numpy()
    Xg = p_flat[:, :Ee * C].reshape(gy, gx, Ee, C).permute(2, 3, 0, 1).double().numpy()
    nan_g = (t_flat[:, Ee * C:].reshape(gy, gx, C).permute(2, 0, 1) > 0).numpy()
    v = (valid_cell.view(gy, gx).numpy()[None] & ~nan_g).astype(float)
    s = SIG_KM / GRID_KM
    gam = alpha / (Ee * (Ee - 1)) + (1 - alpha) / Ee**2 if Ee > 1 else 0.0
    out = []
    for k in range(C):
        blur = lambda f: gaussian_filter(f, s, mode="constant", cval=0.0, truncate=3.0)  # noqa: E731
        den = blur(v[k])
        st = blur(Yg[k] * v[k]) / np.maximum(den, 1e-6)
        sp = [blur(np.where(v[k] > 0, Xg[m, k], Yg[k]) * v[k]) / np.maximum(den, 1e-6) for m in range(Ee)]
        crps = np.mean([np.abs(p - st) for p in sp], 0)
        for i in range(Ee):
            for j in range(i + 1, Ee):
                crps = crps - gam * np.abs(sp[i] - sp[j])
        msk = (v[k] > 0) & (den >= min_cov)
        out.append(crps[msk].mean() / st[msk].std(ddof=1))
    return torch.tensor(out)


hole = torch.from_numpy(~((lat.ravel() > 66) & (lon.ravel() > 22)))
Yn = Y.clone()
idx = torch.randperm(ny * nx, generator=g)[:80]
Yn[idx, 1] = float("nan")
with torch.no_grad():
    for name, (y, x, c) in {"full domain": (Y, X, coords),
                            "hole + missing values": (Yn[hole], X[:, hole], coords[hole])}.items():
        a = loss_chs(y, x, c).double()
        r = reference(y, x, c)
        assert torch.allclose(a, r, rtol=1e-4), (name, a, r)
        print(f"1) {name}: {a.numpy()} = reference {r.numpy()}")


# 2) no block lattice ---------------------------------------------------------------------------
def blocky(offset, block=32, amp=0.3, seed=3):
    gg = torch.Generator().manual_seed(seed)
    vals = amp * torch.randn(E, ny // block + 2, nx // block + 2, C, generator=gg)
    iy = (torch.arange(ny) + (block - offset)) // block
    ix = (torch.arange(nx) + (block - offset)) // block
    return vals[:, iy][:, :, ix].reshape(E, -1, C)


with torch.no_grad():
    ratios = []
    for seed in range(6):
        a = loss_chs(Y, Y.unsqueeze(0) + blocky(0, seed=seed), coords).mean()
        b = loss_chs(Y, Y.unsqueeze(0) + blocky(11, seed=seed), coords).mean()
        ratios.append(float(a / b))
    r = float(np.exp(np.mean(np.log(ratios))))
    assert 0.8 < r < 1.25, ratios
    print(f"2) blocks with edges on a 32-point lattice vs shifted by 11 points: cost ratio {r:.3f}")

# 3) dimensionless --------------------------------------------------------------------------------
with torch.no_grad():
    a = loss_chs(Y, X, coords)
    b = loss_chs(Y * 7.0, X * 7.0, coords)
    assert torch.allclose(a, b, rtol=1e-4), (a, b)
print("3) values x7: same loss")

# 4) large-scale anchor -------------------------------------------------------------------------
with torch.no_grad():
    sd = Y.std(0)
    vals = [float(loss_chs(Y, (Y + k * sd).unsqueeze(0).repeat(E, 1, 1), coords).mean())
            for k in (0.0, 0.1, 0.3, 1.0)]
    assert vals[0] < 1e-6 and vals[0] < vals[1] < vals[2] < vals[3], vals
print("4) members shifted together by 0, 0.1, 0.3, 1 std: " + ", ".join(f"{v:.4f}" for v in vals))

# 5) one member ---------------------------------------------------------------------------------
with torch.no_grad():
    a = loss_chs(Y, X[:1], coords).double()
    r = reference(Y, X[:1], coords)
    assert torch.allclose(a, r, rtol=1e-4), (a, r)
print("5) one member: absolute error of the smoothed fields, as the reference")

# 6) missing values: gradients -------------------------------------------------------------------
Xg = X.clone().requires_grad_(True)
Ynan = Y.clone()
Ynan[idx, :] = float("nan")
loss, _ = lf.global_smooth_crps(Ynan, Xg, coords, None, None, **KW)
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(Xg.grad).all()
assert Xg.grad[:, idx].abs().max() == 0.0 and Xg.grad.abs().sum() > 0
with torch.no_grad():
    full = loss_chs(Y, X, coords)
    holed = loss_chs(Y[hole], X[:, hole], coords[hole])
    rr = float((holed / full).max())
    assert rr < 1.3, rr
print(f"6) missing values: finite, gradients 0 at the missing points; hole / full domain {rr:.3f}")
print("ALL SMOOTH CRPS TESTS PASSED")
