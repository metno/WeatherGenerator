#!/usr/bin/env python3
"""Visualise the latent noise of fe_noise_injection over the MEPS domain.

Uses the model's own NoiseInjector (src/weathergen/model/noise.py). The raw noise field (one
channel, before the conditioning MLP) is drawn at the HEALPix cell centres of the latent level,
exactly as the forecasting engine sees it, and painted back onto the MEPS x/y grid.

Rows: noise_type "iid" and "fourier" for several length scales. Columns: three ensemble members
and the difference of the first two. The bottom panel compares the empirical correlation versus
distance (many draws) with the theoretical squared-exponential kernel.

Example:
    PYTHONPATH=src python scripts/plot_fourier_noise.py --out fourier_noise.png
"""

# ruff: noqa: T201

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from astropy_healpix import HEALPix
from numpy.typing import NDArray

from weathergen.model.noise import EARTH_RADIUS_KM, NoiseInjector

DEFAULT_GRID = (
    "/home/olave/Documents/WG/weathergen-experiments/wajusvle/results/inference/targets/"
    "MEPS/xy/target_2023-06-01T12_wajusvle.nc"
)
ROWS = [
    ("iid (old)", "iid", [1.0]),
    ("fourier, l = 50 km", "fourier", [50.0]),
    ("fourier, l = 150 km", "fourier", [150.0]),
    ("fourier, l = 500 km", "fourier", [500.0]),
    ("fourier, l = [150, 500] km (default)", "fourier", [150.0, 500.0]),
]


def grid_cells(grid_file: str, level: int, stride: int):
    """HEALPix cells covering the (subsampled) MEPS grid, and the grid -> cell index map."""
    ds = xr.open_dataset(grid_file)
    lat = ds.latitude.values[::stride, ::stride]
    lon = ds.longitude.values[::stride, ::stride]
    hp = HEALPix(nside=2**level, order="nested")
    pix = hp.lonlat_to_healpix(lon.ravel() * u.deg, lat.ravel() * u.deg)
    cells, inverse = np.unique(pix, return_inverse=True)
    clon, clat = hp.healpix_to_lonlat(cells)
    # (lat, lon) in radians, the layout of the model's rope_coords
    coords = torch.tensor(np.stack([clat.rad, clon.rad], axis=-1), dtype=torch.float32)
    return coords.unsqueeze(0), inverse.reshape(lat.shape), len(cells)


def draw(noise_type: str, scales: list[float], coords, num_cells: int, num_draws: int, chunk=50):
    """Raw noise, channel 0: (num_draws, num_cells)."""
    inj = NoiseInjector(1, 8, noise_type=noise_type, length_scales_km=scales)
    out = []
    with torch.no_grad():
        for start in range(0, num_draws, chunk):
            n = min(chunk, num_draws - start)
            if noise_type == "fourier":
                out.append(inj.smooth_noise(torch.zeros(n, num_cells, 1), coords)[..., 0])
            else:
                out.append(torch.randn(n, num_cells))
    return torch.cat(out).numpy()


def chord_km(coords: torch.Tensor, i: int) -> NDArray[np.float64]:
    lat, lon = coords[0, :, 0].double(), coords[0, :, 1].double()
    xyz = torch.stack(
        [torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)], -1
    )
    return (EARTH_RADIUS_KM * (xyz - xyz[i]).norm(dim=-1)).numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--grid", default=DEFAULT_GRID, help="MEPS x/y NetCDF with latitude/longitude")
    p.add_argument("--level", type=int, default=8, help="HEALPix level of the latent cells")
    p.add_argument("--stride", type=int, default=3, help="subsample the 2.5 km MEPS grid")
    p.add_argument("--draws", type=int, default=2000, help="draws for the correlation curves")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="fourier_noise.png")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    coords, inverse, num_cells = grid_cells(args.grid, args.level, args.stride)
    print(f"level {args.level}: {num_cells} cells over the MEPS grid")

    # reference cell in the middle of the domain for the correlation curves
    ref = inverse[inverse.shape[0] // 2, inverse.shape[1] // 2]
    dist = chord_km(coords, ref)
    bins = np.linspace(0, 1500, 31)
    centres = 0.5 * (bins[1:] + bins[:-1])

    fig = plt.figure(figsize=(17, 4 * len(ROWS) + 5))
    gs = fig.add_gridspec(len(ROWS) + 1, 4, height_ratios=[1] * len(ROWS) + [1.1])
    ax_corr = fig.add_subplot(gs[-1, :])

    for r, (title, noise_type, scales) in enumerate(ROWS):
        samples = draw(noise_type, scales, coords, num_cells, 3)
        panels = [*samples, samples[0] - samples[1]]
        names = ["member 0", "member 1", "member 2", "member 0 - member 1"]
        for c, (field, name) in enumerate(zip(panels, names, strict=True)):
            ax = fig.add_subplot(gs[r, c])
            lim = 3.0 if c < 3 else 4.0
            im = ax.imshow(field[inverse], origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title(f"{title}\n{name}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.8)

        many = draw(noise_type, scales, coords, num_cells, args.draws)
        corr = (many * many[:, [ref]]).mean(0) / (many.std(0) * many[:, ref].std())
        emp = [
            corr[(dist >= a) & (dist < b)].mean() for a, b in zip(bins[:-1], bins[1:], strict=True)
        ]
        (line,) = ax_corr.plot(centres, emp, "o", ms=4, label=f"{title} (empirical)")
        if noise_type == "fourier":
            d = np.linspace(0, 1500, 300)
            theory = np.mean([np.exp(-(d**2) / (2 * s**2)) for s in scales], axis=0)
            ax_corr.plot(d, theory, "-", color=line.get_color(), alpha=0.7)

    cell_km = np.sqrt(4 * np.pi * EARTH_RADIUS_KM**2 / (12 * 4**args.level))
    ax_corr.axvline(
        cell_km, color="k", ls=":", label=f"hl{args.level} cell size (~{cell_km:.0f} km)"
    )
    ax_corr.axvline(cell_km * 2**3, color="grey", ls=":", label="hl5 cell size")
    ax_corr.axhline(0, color="k", lw=0.5)
    ax_corr.set_xlabel("distance from reference cell (km)")
    ax_corr.set_ylabel("correlation")
    ax_corr.set_title("Correlation vs distance: dots = measured from draws, lines = theory kernel")
    ax_corr.legend(fontsize=9, ncol=2)

    fig.tight_layout()
    fig.savefig(args.out, dpi=70)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
