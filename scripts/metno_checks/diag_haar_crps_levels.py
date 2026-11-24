"""Per-level diagnostic of the Haar CRPS and Haar MSE losses on an inference / validation output.

For every channel and every Haar level l it prints:
  s_l        RMS of the target detail coefficients (LH, HL, HH) at level l, over valid
             positions: the typical size of the coefficients at that scale
  crps       the level loss L_l exactly as in global_haar_wavelet_reshape_varweighted_crps
             (inverse-variance weighted mean over positions, mean over the 3 bands)
  share      L_l / sum_l L_l: the share of each level in the CRPS loss as it is now
  share_n    the same share if each level were divided by s_l (proposed level_norm)
  mse        the level loss M_l of global_haar_wavelet_reshape_varweighted (Haar MSE of the
             ensemble mean, same weights), mean over the 3 bands
  mshare     M_l / sum_l M_l: the share of each level in the Haar MSE as it is now
  mshare_n   the same share if each level were divided by s_l**2 (proposed level_norm)
  flat_w     fraction of valid positions where the target detail energy is < eps, i.e.
             where the weight w = 1/(energy + eps) is ~flat (no relative weighting)
  energy     RMS of the member coefficients / s_l (1 = members as energetic as the target)
  spr/skl    spread / skill of the coefficients: sqrt(unbiased ensemble variance) /
             sqrt(E/(E+1) * mean (ens. mean - target)^2); about 1 for a calibrated ensemble

Values are normalised per channel with the dataset statistics (mean, stdev), as the loss
sees them. No random offset (as in validation).

Usage (from the repository root, on a login or compute node):
  uv run --offline python scripts/metno_checks/diag_haar_crps_levels.py RUN_ID \
      [--mini-epoch 0] [--rank 0] [--samples 8] [--stream CARRA2] \
      [--template .../carra2_rhombus_template.npz] \
      [--stats /e/data1/slmet/CARRA2/carra2-nordic-2016-2025-6h-v1.zarr] \
      [--levels 5] [--eps 0.1] [--alpha 0.95]
"""

import argparse
import math

import numpy as np
import torch

import weathergen.train.loss_modules.loss_functions as lf
from weathergen.common.config import get_model_results
from weathergen.common.io import zarrio_reader

p = argparse.ArgumentParser()
p.add_argument("run_id")
p.add_argument("--mini-epoch", type=int, default=0)
p.add_argument("--rank", type=int, default=0)
p.add_argument("--fstep", type=int, default=None, help="forecast step (default: first)")
p.add_argument("--samples", type=int, default=8, help="number of samples to use (0 = all)")
p.add_argument("--stream", default="CARRA2")
p.add_argument("--template", default="/e/home/jusers/lussana1/jupiter/templates/carra2_rhombus_template.npz")
p.add_argument("--stats", default="/e/data1/slmet/CARRA2/carra2-nordic-2016-2025-6h-v1.zarr",
               help="anemoi dataset with the normalisation statistics ('' = per-sample target std)")
p.add_argument("--levels", type=int, default=5)
p.add_argument("--eps", type=float, default=0.1)
p.add_argument("--alpha", type=float, default=0.95)
args = p.parse_args()
L, eps, alpha = args.levels, args.eps, args.alpha


def haar(F):
    a, b, c, d = F[..., 0::2, 0::2], F[..., 0::2, 1::2], F[..., 1::2, 0::2], F[..., 1::2, 1::2]
    return (a + b + c + d) / 2, ((a + c) - (b + d)) / 2, ((a + b) - (c + d)) / 2, ((a + d) - (b + c)) / 2


def load_stats(path, channels):
    if not path:
        return None
    try:
        from anemoi.datasets import open_dataset
        ds = open_dataset(path)
        idx = [ds.variables.index(ch) for ch in channels]
        return np.asarray(ds.statistics["mean"])[idx], np.asarray(ds.statistics["stdev"])[idx]
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not read statistics from {path} ({e}); "
              f"using the per-sample target mean/std instead")
        return None


fname = get_model_results(args.run_id, args.mini_epoch, args.rank)
print(f"results: {fname}")


def _owner():
    pass


_owner.__name__ = "diag_haar_crps_levels"
tmpl, (ny, nx) = lf._load_haar_template(_owner, args.template, args.stream, "diag")
assert tmpl is not None, f"cannot read template {args.template}"

acc = None  # accumulated sums per (channel, level)
n_used = 0
with zarrio_reader(fname) as zio:
    samples = sorted(int(s) for s in zio.samples)
    if args.samples > 0:
        samples = samples[: args.samples]
    fstep = args.fstep if args.fstep is not None else sorted(int(f) for f in zio.forecast_steps)[0]
    stats = None
    for s in samples:
        item = zio.get_data(s, args.stream, fstep)
        if item.target is None or item.prediction is None:
            continue
        y = np.asarray(item.target.data, dtype=np.float64)          # (P, C)
        x = np.asarray(item.prediction.data, dtype=np.float64)      # (P, C, E)
        if y.shape[0] == 0:
            continue
        chans = list(item.target.channels)
        coords = torch.tensor(np.asarray(item.target.coords), dtype=torch.float32)
        if stats is None and n_used == 0:
            stats = load_stats(args.stats, chans)
            print(f"channels: {chans}; members: {x.shape[2]}; samples: {len(samples)}; "
                  f"normalisation: {'dataset statistics' if stats is not None else 'per-sample'}")
        if stats is not None:
            mu, sd = stats
        else:
            mu, sd = np.nanmean(y, 0), np.nanstd(y, 0)
        y = (y - mu) / sd
        x = (x - mu[None, :, None]) / sd[None, :, None]
        P, C = y.shape
        E = x.shape[2]
        yt = torch.from_numpy(y)
        xt = torch.from_numpy(x).permute(2, 0, 1)                    # (E, P, C)

        # regrid exactly as the loss (nearest, valid mask, members := target on invalid)
        nan_t = torch.isnan(yt)
        tgt = torch.where(nan_t, torch.zeros_like(yt), yt)
        prd = torch.where(nan_t.unsqueeze(0), torch.zeros_like(xt), xt)
        p_cols = prd.permute(1, 0, 2).reshape(P, E * C).float()
        t_cols = tgt.repeat(1, E).float()
        nan_ind = nan_t.float()
        t_flat, p_flat, valid_cell, empty = lf._reshape_regrid_to_grid(
            torch.cat([t_cols, nan_ind], 1), torch.cat([p_cols, nan_ind], 1), coords, tmpl,
            ny, nx, E * C + C, torch.device("cpu"), stream_name=args.stream, _report_owner=_owner)
        if empty:
            print(f"sample {s}: template does not cover the data; skipped")
            continue
        Y = t_flat[:, :C].reshape(ny, nx, C).permute(2, 0, 1).double()
        X = p_flat[:, : E * C].reshape(ny, nx, E, C).permute(2, 3, 0, 1).double()
        nan_g = t_flat[:, E * C:].reshape(ny, nx, C).permute(2, 0, 1) > 0
        valid = valid_cell.view(1, ny, nx) & ~nan_g
        X = torch.where(valid.unsqueeze(0), X, Y.unsqueeze(0))

        f = 2**L
        Ny, Nx = math.ceil(ny / f) * f, math.ceil(nx / f) * f
        Yp = torch.zeros(C, Ny, Nx, dtype=torch.double)
        Yp[:, :ny, :nx] = Y
        Xp = torch.zeros(E, C, Ny, Nx, dtype=torch.double)
        Xp[..., :ny, :nx] = X
        Vp = torch.zeros(C, Ny, Nx, dtype=torch.double)
        Vp[:, :ny, :nx] = valid.double()

        if acc is None:
            keys = ["crps", "mse", "s2", "nv", "flat", "xe2", "spr", "skl"]
            acc = {k: torch.zeros(C, L, dtype=torch.double) for k in keys}
        gE = alpha / (E * (E - 1)) + (1 - alpha) / E**2 if E > 1 else 0.0
        Yl, Xl, Vl = Yp, Xp, Vp
        for lvl in range(L):
            YLL, *Yb = haar(Yl)
            XLL, *Xb = haar(Xl)
            # a coefficient is "valid" if its whole 2x2 block of the previous level is valid
            vb = torch.minimum(torch.minimum(Vl[..., 0::2, 0::2], Vl[..., 0::2, 1::2]),
                               torch.minimum(Vl[..., 1::2, 0::2], Vl[..., 1::2, 1::2]))
            en = Yb[0] ** 2 + Yb[1] ** 2 + Yb[2] ** 2
            w = 1.0 / (en + eps)
            Ll = 0.0
            Ml = 0.0
            for yb, xb in zip(Yb, Xb):
                Ml = Ml + (w * (xb.mean(0) - yb) ** 2).sum((1, 2)) / w.sum((1, 2))
                crps = (xb - yb).abs().mean(0)
                for i in range(E - 1):
                    crps = crps - gE * (xb[i].unsqueeze(0) - xb[i + 1:]).abs().sum(0)
                Ll = Ll + (w * crps).sum((1, 2)) / w.sum((1, 2))
                acc["s2"][:, lvl] += (vb * yb**2).sum((1, 2))
                acc["xe2"][:, lvl] += (vb * (xb**2).mean(0)).sum((1, 2))
                if E > 1:
                    acc["spr"][:, lvl] += (vb * xb.var(0, unbiased=True)).sum((1, 2))
                    acc["skl"][:, lvl] += (vb * (xb.mean(0) - yb) ** 2).sum((1, 2)) * E / (E + 1)
            acc["crps"][:, lvl] += Ll / 3
            acc["mse"][:, lvl] += Ml / 3
            acc["nv"][:, lvl] += 3 * vb.sum((1, 2))
            acc["flat"][:, lvl] += (vb * (en < eps)).sum((1, 2))
            Yl, Xl, Vl = YLL, XLL, vb
        n_used += 1

assert n_used > 0, "no usable samples"
crps = acc["crps"] / n_used
s_l = (acc["s2"] / acc["nv"]).sqrt()
share = crps / crps.sum(1, keepdim=True)
cn = crps / s_l
share_n = cn / cn.sum(1, keepdim=True)
mse = acc["mse"] / n_used
mshare = mse / mse.sum(1, keepdim=True)
mn = mse / s_l**2
mshare_n = mn / mn.sum(1, keepdim=True)
flat = acc["flat"] / (acc["nv"] / 3)
energy = (acc["xe2"] / acc["nv"]).sqrt() / s_l
ratio = (acc["spr"] / acc["skl"].clamp(min=1e-30)).sqrt()

km = [f"{2.5 * 2**l:g}-{2.5 * 2**(l + 1):g} km" for l in range(L)]
print(f"\n{n_used} samples, levels: " + ", ".join(f"L{l} {km[l]}" for l in range(L)))
cols = ["s_l", "crps", "share", "share_n", "mse", "mshare", "mshare_n", "flat_w", "energy", "spr/skl"]
for c, ch in enumerate(chans):
    print(f"\n{ch}")
    print("  level " + "".join(f"{h:>10}" for h in cols))
    for lvl in range(L):
        vals = [s_l[c, lvl], crps[c, lvl], share[c, lvl], share_n[c, lvl], mse[c, lvl],
                mshare[c, lvl], mshare_n[c, lvl], flat[c, lvl],
                energy[c, lvl], ratio[c, lvl]]
        print(f"  L{lvl}    " + "".join(f"{float(v):10.4f}" for v in vals))

print("\nmean over channels")
print("  level " + "".join(f"{h:>10}" for h in cols))
for lvl in range(L):
    vals = [s_l[:, lvl].mean(), crps[:, lvl].mean(), share[:, lvl].mean(), share_n[:, lvl].mean(),
            mse[:, lvl].mean(), mshare[:, lvl].mean(), mshare_n[:, lvl].mean(),
            flat[:, lvl].mean(), energy[:, lvl].mean(), ratio[:, lvl].mean()]
    print(f"  L{lvl}    " + "".join(f"{float(v):10.4f}" for v in vals))
print("\nshare / mshare: share of each level in the Haar CRPS / Haar MSE now; "
      "share_n / mshare_n: if each level were divided by s_l / s_l**2")
