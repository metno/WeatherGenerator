#!/usr/bin/env python
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
RoPE wavelength advisor for regional / multi-resolution configs.

Builds the latent domains of a config exactly as the model does (Domain.from_level_config),
reproduces the RoPE coordinate frames of the code base in numpy, measures the geometry
that matters for each RoPE, and suggests wavelengths:

  rope_2D_wavelengths          encoder global engine (frame: fe_rope_coords, i.e.
                               equirectangular about the domain centre, in cells of the
                               level). Depends on the DOMAIN: its size (longest) and its
                               latitude span (east-west distortion -> shortest).
  decoder_rope_wavelengths     decoder cross-attention point -> latent cells (frame:
  (or per stream                dec_rope_frame, azimuthal equidistant, in cells of the
   target_readout.               stream's level). Depends on the DATA SPACING (shortest)
   rope_wavelengths)            and on the level ratio of the decode levels (longest).
  latent_cascade.               fine cell -> 9 coarse cells around its parent, in coarse
    down_rope_wavelengths       cells. Depends on the level ratio only.
  latent_cascade.               coarse cell -> its children, in coarse cells. Level ratio.
    up_rope_wavelengths

Rules used for the suggestions (geometric series between shortest and longest):
  shortest = 2 x (typical = median spacing that must be resolved) -> typical neighbours
             are half a wavelength apart at the fastest frequency (Nyquist)
  longest  = 2 x (largest offset the attention ever sees) -> phase stays < pi at the
             slowest frequency, so the largest offsets are still unambiguous
The encoder frame is not distance-preserving: the script reports how much it stretches or
squeezes neighbour distances; that cannot be fixed by the wavelengths, only by the frame.

Usage (from the repository root, in the training environment):

  python scripts/rope_advisor.py --config config/config_e01_ae.yml \\
      --stream carra2:8:/e/project1/.../carra2_rhombus_domain.npz \\
      --stream era5:5:31

  --stream NAME:LEVEL:SPACING[:DECODE_LEVELS]
      LEVEL         encode level of the stream (its target cells)
      SPACING       grid spacing in km, or a points file (.npz / .nc / zarr) from which the
                    median nearest-neighbour spacing inside the domain is measured
      DECODE_LEVELS optional, comma separated (default: every ladder level <= LEVEL,
                    which is what stream_levels assigns to a fine stream)

Run it on an old and a new config to compare domains.
"""

import argparse
import math
import os
import sys
import warnings

import numpy as np
import yaml

R_EARTH_KM = 6371.0


# --------------------------------------------------------------------------------------
# imports from the repository (light: domain.py only needs numpy + astropy_healpix)
# --------------------------------------------------------------------------------------
def _import_domain(repo: str | None):
    if repo is not None:
        sys.path.insert(0, os.path.join(os.path.abspath(repo), "src"))
    try:
        from weathergen.datasets import domain as dom_mod
    except ImportError as e:
        raise SystemExit(
            "Cannot import weathergen.datasets.domain. Run inside the training environment "
            "or pass --repo /path/to/WeatherGenerator."
        ) from e
    return dom_mod


# --------------------------------------------------------------------------------------
# geometry (numpy replicas of the torch code, float64)
# --------------------------------------------------------------------------------------
def cell_rad(level: int) -> float:
    """Mean cell size [rad], as in fe_rope_coords / dec_rope_frame."""
    return math.sqrt(4.0 * math.pi / (12 * 4**level))


def cell_km(level: int) -> float:
    return cell_rad(level) * R_EARTH_KM


def centres_r3(level: int, cells: np.ndarray, dx=0.5, dy=0.5) -> np.ndarray:
    """healpix_verts_rots(level, dx, dy)[0][cells], unit vectors (n, 3)."""
    import astropy_healpix as hp

    lons, lats = hp.healpix_to_lonlat(np.asarray(cells), 2**level, dx=dx, dy=dy, order="nested")
    th = np.pi / 2.0 - lats.value
    ph = lons.value
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1)


def enc_coords(level: int, cells: np.ndarray) -> np.ndarray:
    """model.fe_rope_coords: (y, x) in cells, equirectangular about the domain centre."""
    v = centres_r3(level, cells)
    lat = np.arctan2(v[:, 2], np.hypot(v[:, 0], v[:, 1]))
    lon = np.arctan2(v[:, 1], v[:, 0])
    lon_c = np.arctan2(np.sin(lon).mean(), np.cos(lon).mean())
    dlon = np.remainder(lon - lon_c + np.pi, 2 * np.pi) - np.pi
    lat_c = 0.5 * (lat.max() + lat.min())
    c = cell_rad(level)
    return np.stack([(lat - lat_c) / c, dlon * np.cos(lat_c) / c], axis=-1)


def aeqd_frame(level: int, cells: np.ndarray) -> dict:
    """positional_encoding.dec_rope_frame."""
    v = centres_r3(level, cells)
    c0 = v.mean(0)
    c0 /= np.linalg.norm(c0)
    east = np.cross([0.0, 0.0, 1.0], c0)
    if np.linalg.norm(east) < 1e-6:
        raise SystemExit("domain centred on a pole: dec_rope_frame does not support this")
    east /= np.linalg.norm(east)
    north = np.cross(c0, east)
    return dict(c0=c0, east=east, north=north, cell=cell_rad(level))


def aeqd_project(v: np.ndarray, fr: dict) -> np.ndarray:
    """positional_encoding.dec_rope_project: (y, x) in cells of the frame's level."""
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    cosd = v @ fr["c0"]
    t = v - cosd[:, None] * fr["c0"]
    sind = np.linalg.norm(t, axis=-1)
    theta = np.arctan2(sind, cosd)
    scale = np.where(sind > 1e-12, theta / np.maximum(sind, 1e-12), 1.0)
    return np.stack(
        [scale * (t @ fr["north"]) / fr["cell"], scale * (t @ fr["east"]) / fr["cell"]], axis=-1
    )


def edge_neighbour_pairs(domain) -> tuple[np.ndarray, np.ndarray]:
    """(i, j) compact index pairs of active cells that share an EDGE (the 4 closest of the 8
    healpix neighbours, by true distance)."""
    nb = domain.neighbours_compact()  # (n, 8), self where missing / outside
    n = len(domain)
    v = centres_r3(domain.healpix_level, domain.active_cells)
    d = np.linalg.norm(v[nb] - v[:, None, :], axis=-1)  # (n, 8)
    d[nb == np.arange(n)[:, None]] = np.inf
    k = np.argsort(d, axis=1)[:, :4]
    i = np.repeat(np.arange(n), 4)
    j = nb[np.arange(n)[:, None], k].ravel()
    ok = np.isfinite(d[np.arange(n)[:, None], k].ravel())
    return i[ok], j[ok]


def parent_compact(dom_f, dom_c) -> np.ndarray:
    """Compact index (in dom_c) of the parent of every active fine cell, -1 if inactive."""
    shift = 2 * (dom_f.healpix_level - dom_c.healpix_level)
    return dom_c.to_compact_safe(dom_f.active_cells >> shift)


def cell_radius(level: int, cells: np.ndarray, fr: dict) -> float:
    """Largest distance cell centre -> cell corner in the frame (a point inside a cell is at
    most this far from its centre)."""
    ctr = aeqd_project(centres_r3(level, cells), fr)
    r = 0.0
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        cor = aeqd_project(centres_r3(level, cells, dx, dy), fr)
        r = max(r, float(np.linalg.norm(cor - ctr, axis=-1).max()))
    return r


def data_spacing_km(dom_mod, path: str, domain) -> tuple[float, int]:
    """Median nearest-neighbour distance [km] of the points of `path` inside `domain`."""
    from scipy.spatial import cKDTree

    lats, lons = dom_mod._load_points(os.path.abspath(path))
    keep = domain.point_mask(lats, lons)
    lats, lons = lats[keep], lons[keep]
    if len(lats) < 2:
        raise SystemExit(f"{path}: fewer than 2 points inside the domain")
    la, lo = np.radians(lats), np.radians(lons)
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)
    tree = cKDTree(xyz)
    rng = np.random.default_rng(0)
    q = xyz[rng.choice(len(xyz), size=min(len(xyz), 50_000), replace=False)]
    d, _ = tree.query(q, k=2)
    return float(np.median(2.0 * np.arcsin(np.clip(d[:, 1] / 2.0, 0, 1))) * R_EARTH_KM), len(lats)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def nice_up(x: float) -> float:
    """Round up to 2 significant digits (1.37 -> 1.4, 23.1 -> 24)."""
    if x <= 0:
        return x
    e = math.floor(math.log10(x)) - 1
    return round(math.ceil(x / 10**e - 1e-9) * 10**e, 6)


def fmt(x) -> str:
    return f"{x:.3g}" if isinstance(x, float) else str(x)


def verdict(cur, rec_lo, rec_hi, auto_hi=None) -> str:
    """Compare a configured [lo, hi] with the suggestion."""
    if cur is None:
        return "not set in config"
    lo, hi = float(cur[0]), float(cur[1])
    notes = []
    if lo > 1.6 * rec_lo:
        notes.append(f"shortest {lo:g} > 1.5x suggested: finest offsets under-resolved")
    elif lo < rec_lo / 1.5:
        notes.append(f"shortest {lo:g} < suggested/1.5: frequencies spent on unresolved scales")
    hi_eff = auto_hi if (hi <= 0 and auto_hi is not None) else hi
    if hi_eff < rec_hi / 1.25:
        notes.append(f"longest {hi_eff:.3g} < suggested: large offsets wrap (aliasing)")
    elif hi_eff > 4 * rec_hi:
        notes.append(f"longest {hi_eff:.3g} >> suggested: frequencies spent beyond the range used")
    return "OK" if not notes else "CHECK: " + "; ".join(notes)


def pct(a, q):
    return float(np.percentile(a, q))


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True, help="model config (yml)")
    ap.add_argument("--repo", default=None, help="WeatherGenerator checkout (if not installed)")
    ap.add_argument("--stream", action="append", default=[], help="NAME:LEVEL:SPACING[:DECODE_LEVELS]")
    args = ap.parse_args()

    dom_mod = _import_domain(args.repo)
    import logging

    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")

    with open(args.config) as f:
        cf = yaml.safe_load(f)

    # ---- domains, exactly as the model builds them
    if cf.get("latent_levels"):
        level_cfg = {int(e["level"]): e.get("domain", None) for e in cf["latent_levels"]}
    else:
        level_cfg = {int(cf["healpix_level"]): cf.get("domain", None)}
    ladder = sorted(level_cfg)
    doms = {L: dom_mod.Domain.from_level_config(L, level_cfg[L]) for L in ladder}

    print(f"\nconfig: {args.config}")
    print(f"latent levels: {ladder}\n")

    # ================================================================ encoder
    rope_cfg = cf.get("rope_2D_wavelengths", None)
    print("=" * 88)
    print("ENCODER  rope_2D_wavelengths  (frame fe_rope_coords: equirectangular, cells of level)")
    print(f"  config: {rope_cfg}   (rope_2D={cf.get('rope_2D')}, rope_2D_coords={cf.get('rope_2D_coords')})")
    print("=" * 88)
    enc_lo_all, enc_rows, enc_dist = [], [], []
    for L in ladder:
        d = doms[L]
        if d.is_global:
            print(f"  hl{L}: global domain -- the equirectangular frame is not meaningful; skipped")
            continue
        xy = enc_coords(L, d.active_cells)
        span = float((xy.max(0) - xy.min(0)).max())
        i, j = edge_neighbour_pairs(d)
        d_enc = np.linalg.norm(xy[i] - xy[j], axis=-1)
        fr = aeqd_frame(L, d.active_cells)
        xy_true = aeqd_project(centres_r3(L, d.active_cells), fr)
        d_true = np.linalg.norm(xy_true[i] - xy_true[j], axis=-1)
        ratio = d_enc / d_true
        v = centres_r3(L, d.active_cells)
        lat = np.degrees(np.arctan2(v[:, 2], np.hypot(v[:, 0], v[:, 1])))
        rec_lo = nice_up(2.0 * pct(d_true, 50))
        lo_cfg = float(rope_cfg[0]) if rope_cfg else rec_lo
        auto_hi = max(2.0 * span, 4.0 * lo_cfg)
        rec_hi = nice_up(2.0 * span)
        stretched = float(np.mean(ratio > 1.5))
        squeezed = float(np.mean(ratio < 1 / 1.5))
        enc_lo_all.append(rec_lo)
        enc_dist.append((L, pct(ratio, 1), pct(ratio, 99), stretched, squeezed))
        print(f"\n  hl{L}: {len(d)} cells ({cell_km(L):.1f} km), cell centres lat "
              f"{lat.min():.1f}..{lat.max():.1f} deg")
        print(f"    extent (y, x) in cells        : {np.ptp(xy[:, 0]):.1f} x {np.ptp(xy[:, 1]):.1f}"
              f"   -> longest=0 resolves to {auto_hi:.1f} cells")
        print(f"    edge-neighbour distance, true  : p1 {pct(d_true, 1):.2f}  median {pct(d_true, 50):.2f}"
              f"  p99 {pct(d_true, 99):.2f} cells")
        print(f"    edge-neighbour distance, frame : p1 {pct(d_enc, 1):.2f}  median {pct(d_enc, 50):.2f}"
              f"  p99 {pct(d_enc, 99):.2f}  max {d_enc.max():.2f} cells")
        print(f"    frame / true distance          : p1 {pct(ratio, 1):.2f}  median {pct(ratio, 50):.2f}"
              f"  p99 {pct(ratio, 99):.2f}  (1 = undistorted); "
              f"{100 * stretched:.1f}% of pairs stretched >1.5x, {100 * squeezed:.1f}% squeezed <1/1.5")
        print(f"    suggested: [{rec_lo:g}, 0]  (0 -> auto {rec_hi:g})")
        enc_rows.append((L, rec_lo))
        if pct(ratio, 99) > 1.5 or pct(ratio, 1) < 1 / 1.5:
            print("    NOTE: the equirectangular frame (x scaled by cos of the centre latitude) "
                  "distorts east-west distances here; nearby cells in the stretched part look "
                  "farther apart to the attention than they are. Wavelengths cannot fix this.")
    if enc_lo_all:
        rec = [max(enc_lo_all), 0.0]
        print(f"\n  >> rope_2D_wavelengths: {rec}   (one value for all levels; longest auto per level)")
        if rope_cfg is None:
            print("     not set in config (original 2*pi*10000^(i/n) series is used)")
        else:
            lo, hi = float(rope_cfg[0]), float(rope_cfg[1])
            notes = []
            if lo < rec[0] / 1.25:
                notes.append(f"shortest {lo:g} < suggested {rec[0]:g}: neighbour cells alias "
                             "at the fastest frequency where the frame is stretched")
            elif lo > 1.5 * rec[0]:
                notes.append(f"shortest {lo:g} > 1.5x suggested: adjacent cells less distinct")
            if hi > 0:
                notes.append(f"longest fixed at {hi:g}: compare with the per-level auto values "
                             "above (0 adapts to the domain)")
            print("     " + ("OK" if not notes else "CHECK: " + "; ".join(notes)))
        worst = max(enc_dist, key=lambda t: max(t[2], 1 / t[1]))
        print(f"     frame distortion (worst level hl{worst[0]}): neighbour distances x{worst[1]:.2f} .. "
              f"x{worst[2]:.2f} (p1..p99)")

    # ================================================================ decoder
    dec_cfg = cf.get("decoder_rope_wavelengths", [0.25, 64.0] if cf.get("decoder_rope_2D") else None)
    print("\n" + "=" * 88)
    print("DECODER  decoder_rope_wavelengths  (frame dec_rope_frame: azimuthal equidistant,")
    print("         cells of the stream's own level)")
    print(f"  config: {dec_cfg}   (decoder_rope_2D={cf.get('decoder_rope_2D')})")
    print("=" * 88)
    if not args.stream:
        print("  no --stream given: pass e.g. --stream carra2:8:/path/template.npz --stream era5:5:31")
    dec_los, dec_his = [], []
    for s in args.stream:
        parts = s.split(":")
        if len(parts) < 3:
            raise SystemExit(f"--stream {s!r}: expected NAME:LEVEL:SPACING[:DECODE_LEVELS]")
        name, Ls = parts[0], int(parts[1])
        if Ls not in doms:
            raise SystemExit(f"--stream {s!r}: level {Ls} is not a latent level {ladder}")
        spacing_src = ":".join(parts[2:3])
        dec_lv = (sorted(int(x) for x in parts[3].split(",")) if len(parts) > 3
                  else [L for L in ladder if L <= Ls])
        d = doms[Ls]
        try:
            sp_km, npts = float(spacing_src), None
        except ValueError:
            sp_km, npts = data_spacing_km(dom_mod, spacing_src, d)
        sp_cells = sp_km / cell_km(Ls)
        fr = aeqd_frame(Ls, d.active_cells)
        xy = aeqd_project(centres_r3(Ls, d.active_cells), fr)
        nb9 = np.concatenate([np.arange(len(d))[:, None], d.neighbours_compact()], axis=1)
        offs = [np.linalg.norm(xy[nb9] - xy[:, None, :], axis=-1).max(1)]
        for Lc in [L for L in dec_lv if L < Ls]:
            dc = doms[Lc]
            xy_c = aeqd_project(centres_r3(Lc, dc.active_cells), fr)
            nb9c = np.concatenate([np.arange(len(dc))[:, None], dc.neighbours_compact()], axis=1)
            par = parent_compact(d, dc)
            ring = np.where((par >= 0)[:, None, None], xy_c[nb9c[np.maximum(par, 0)]], xy[:, None, :])
            offs.append(np.linalg.norm(ring - xy[:, None, :], axis=-1).max(1))
        r_in = cell_radius(Ls, d.active_cells, fr)
        max_off = float(np.max(offs)) + r_in
        rec_lo = nice_up(2.0 * sp_cells)
        rec_hi = nice_up(2.0 * max_off)
        dec_los.append(rec_lo)
        dec_his.append(rec_hi)
        src = f"{npts} points in domain" if npts else "given"
        print(f"\n  {name}: level hl{Ls} ({cell_km(Ls):.1f} km cells), decode levels {dec_lv}")
        print(f"    data spacing: {sp_km:.2f} km ({src}) = {sp_cells:.3f} cells")
        print(f"    largest point -> KV-cell offset: {max_off:.1f} cells "
              f"(ring offsets + point inside cell {r_in:.2f})")
        print(f"    suggested: [{rec_lo:g}, {rec_hi:g}]   current: {verdict(dec_cfg, rec_lo, rec_hi)}")
        print(f"    (per-stream override in the stream yml: target_readout: "
              f"{{rope_wavelengths: [{rec_lo:g}, {rec_hi:g}]}})")
    if dec_los:
        rec = [min(dec_los), max(dec_his)]
        print(f"\n  >> decoder_rope_wavelengths: {rec}   (global: finest shortest, largest longest)")
        print(f"     {verdict(dec_cfg, rec[0], rec[1])}")

    # ================================================================ cascade
    lc_cfg = cf.get("latent_cascade", {}) or {}
    print("\n" + "=" * 88)
    print("LATENT CASCADE  down_rope_wavelengths / up_rope_wavelengths  (dec_rope_frame of the")
    print("                fine domain, in COARSE cells)")
    print(f"  config: down_rope={lc_cfg.get('down_rope', False)} "
          f"{lc_cfg.get('down_rope_wavelengths', [0.25, 8.0])}, "
          f"up_rope={lc_cfg.get('up_rope', False)} {lc_cfg.get('up_rope_wavelengths', [0.25, 4.0])}")
    print("=" * 88)
    for Lc, Lf in zip(ladder[:-1], ladder[1:]):
        dc, df = doms[Lc], doms[Lf]
        scale = 2.0 ** (Lf - Lc)
        fr = aeqd_frame(Lf, df.active_cells)
        xy_f = aeqd_project(centres_r3(Lf, df.active_cells), fr) / scale
        xy_c = aeqd_project(centres_r3(Lc, dc.active_cells), fr) / scale
        par = parent_compact(df, dc)
        ok = par >= 0
        nb9c = np.concatenate([np.arange(len(dc))[:, None], dc.neighbours_compact()], axis=1)
        down = np.linalg.norm(xy_c[nb9c[par[ok]]] - xy_f[ok][:, None, :], axis=-1).max()
        up = np.linalg.norm(xy_f[ok] - xy_c[par[ok]], axis=-1).max()
        i, j = edge_neighbour_pairs(df)
        fine_sp = pct(np.linalg.norm(xy_f[i] - xy_f[j], axis=-1), 50)
        rec_lo = nice_up(2.0 * fine_sp)
        rec_dn, rec_up = [rec_lo, nice_up(2.0 * down)], [rec_lo, nice_up(2.0 * up)]
        print(f"\n  hl{Lc} <-> hl{Lf}: {scale:g}x{scale:g} children per parent, "
              f"{int(ok.sum())}/{len(df)} fine cells with an active parent")
        print(f"    fine-cell spacing (median): {fine_sp:.3f} coarse cells")
        print(f"    down: largest fine -> coarse-ring offset {down:.2f}  suggested {rec_dn}   "
              f"{verdict(lc_cfg.get('down_rope_wavelengths', [0.25, 8.0]), *rec_dn)}")
        print(f"    up  : largest child -> parent offset    {up:.2f}  suggested {rec_up}   "
              f"{verdict(lc_cfg.get('up_rope_wavelengths', [0.25, 4.0]), *rec_up)}")
        if ok.sum() < len(df):
            print(f"    NOTE: {len(df) - int(ok.sum())} fine cells have no active parent: grow the "
                  f"coarse domain (pad_rings) so it covers the fine one")
    print()


if __name__ == "__main__":
    main()
