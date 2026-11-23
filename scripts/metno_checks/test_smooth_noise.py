"""Smooth noise (stochastic.noise_scales_km) on real HEALPix hl8 cells over the CARRA2 area (CPU):
   no blocks, unit variance, correlation length as configured, same parameter shapes as the
   block noise (checkpoint compatible), zero start unchanged."""
import types

import astropy_healpix as hp
import numpy as np
import torch

from weathergen.model.stochastic import LatentNoise


class Cf(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)

    def __getattr__(self, k):
        return self[k]


def cf_(**sc):
    base = dict(enabled=True, level=8, noise_levels=[6, 7], noise_dim=8, smooth_passes=1,
                num_blocks=1, att_dim=32, att_heads=4, members_train=2, members_val=2, seed=7)
    base.update(sc)
    return Cf(ae_global_dim_embed=64, ae_local_num_queries=1, num_register_tokens=0,
              num_class_tokens=0, norm_eps=1e-5, stochastic=base)


# hl8 cells (nested) over 57-68 N, 2-28 E, with 1-ring neighbours in compact indices
nside = 2**8
lon, lat = np.meshgrid(np.linspace(2, 28, 400), np.linspace(57, 68, 300))
import astropy.units as u
cells = np.unique(hp.lonlat_to_healpix(lon.ravel() * u.deg, lat.ravel() * u.deg, nside,
                                       order="nested"))
N = len(cells)
nbr = hp.neighbours(cells, nside, order="nested").T                  # (N, 8)
pos = {c: i for i, c in enumerate(cells)}
nb = np.array([[i] + [pos.get(c, i) for c in row] for i, row in enumerate(nbr)])  # self fallback
nb = torch.as_tensor(nb, dtype=torch.long)
clon, clat = hp.healpix_to_lonlat(cells, nside, order="nested")
clon, clat = np.asarray(clon.to_value(u.rad)), np.asarray(clat.to_value(u.rad))
xyz = np.stack([np.cos(clat) * np.cos(clon), np.cos(clat) * np.sin(clon), np.sin(clat)], 1)

# 1) number of passes per scale
noise = LatentNoise(cf_(noise_scales_km=[50, 80, 120, 160, 200]), 8)
print(f"hl8 cell {noise.cell_km:.1f} km; passes {noise.scale_passes}; {N} cells")
assert noise.scale_passes == [1, 4, 8, 15, 23], noise.scale_passes

# 2) unit variance, no blocks (no repeated values), correlation vs distance
noise = LatentNoise(cf_(noise_scales_km=[50, 120]), 8)
corr_at = {}
zs = [noise._draw_smooth(1, N, nb, torch.Generator().manual_seed(s), "cpu")[0] for s in range(6)]
zs = torch.stack(zs)                                                  # (6, N, 16)
assert torch.allclose(zs.std(1), torch.ones(6, 16), atol=1e-4)
for c in range(8):
    assert len(torch.unique(zs[0, :, c])) > 0.99 * N, "many repeated values: blocks"
rng = np.random.default_rng(0)
i = rng.integers(0, N, 40000); j = rng.integers(0, N, 40000)
dist = 6371.0 * np.arccos(np.clip((xyz[i] * xyz[j]).sum(1), -1, 1))
for si, km in enumerate(noise.noise_scales_km):
    a = zs[:, i, si * 8:(si + 1) * 8].numpy(); b = zs[:, j, si * 8:(si + 1) * 8].numpy()
    prod = (a * b).mean(axis=(0, 2))
    line = []
    for d0 in (25, 50, 100, 150, 250):
        sel = np.abs(dist - d0) < 6
        line.append(f"{d0} km: {prod[sel].mean():+.2f}")
        corr_at[(km, d0)] = prod[sel].mean()
    print(f"scale {km:.0f} km, correlation at " + " | ".join(line))
assert 0.25 < corr_at[(50, 50)] < 0.5, corr_at[(50, 50)]
assert 0.25 < corr_at[(120, 100)] < 0.6 and corr_at[(120, 250)] < 0.15, corr_at
print("2) unit variance, no blocks, correlation about e^-1 at the chosen scale")

# 3) same parameter shapes as the block noise -> checkpoint of the block run loads strictly
old = LatentNoise(cf_(), 8)
new = LatentNoise(cf_(noise_scales_km=[50, 120]), 8)
new.load_state_dict(old.state_dict(), strict=True)
print("3) block-noise checkpoint loads into the smooth-noise model (strict)")

# 4) zero start unchanged
mp_ = types.SimpleNamespace(hp_nbours=nb.int(), domain=types.SimpleNamespace(active_cells=cells))
params = types.SimpleNamespace(params_for=lambda lvl: mp_)
fresh = LatentNoise(cf_(noise_scales_km=[50, 120]), 8)
fresh.train(); fresh.begin_forward()
x = torch.randn(1, N, 64)
assert torch.equal(fresh(x, params, 0), x)
a0 = fresh.draw(1, cells, nb, 0, "cpu"); a1 = fresh.draw(1, cells, nb, 1, "cpu")
assert not torch.equal(a0, a1)
print("4) zero start identical; members get different noise")
print("ALL SMOOTH NOISE TESTS PASSED")
