# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Diagnostic scatter maps of target vs prediction, per stream.

This is a DEBUG tool for verifying that a regional (cropped) domain places data
where it should. It is intentionally standalone and side-effect free: it writes
PNGs to disk and touches nothing in the training path.

Typical use: call once at the first step of a run from
loss_module_physical.compute_loss(), then look at the PNGs and remove the call.

Why here and not in lp_loss(): lp_loss only receives (target, pred). The
coordinates live one level up, in LossPhysical.compute_loss, where
targets_coords_batch[target_idx] is target_coords_raw -- geographic lat/lon in
DEGREES, shape (n_points, 2) as [lat, lon]. That is the only place where
coordinates, targets and predictions are all in scope together.
"""

import logging
from pathlib import Path

import numpy as np
import torch

_logger = logging.getLogger(__name__)

# module-level guard so we plot once per (stream, channel), not once per batch
_PLOTTED: set = set()


def plot_target_pred_map(
    stream_name: str,
    target: torch.Tensor,
    pred: torch.Tensor,
    coords: torch.Tensor,
    channel_names: list[str] | None = None,
    channel_idx: int = 0,
    out_dir: str | Path = "./plots/domain_check",
    tag: str = "",
    max_points: int = 200_000,
    once: bool = True,
) -> None:
    """
    Scatter-plot target and prediction on a lat/lon map for one channel.

    Parameters
    ----------
    stream_name :
        e.g. "ERA5" or "MEPS", used in the filename and title.
    target :
        (n_points, n_channels) tensor of target values (normalised model space).
    pred :
        (ens, n_points, n_channels) or (n_points, n_channels). If an ensemble
        dimension is present it is averaged, matching what lp_loss does.
    coords :
        (n_points, 2) geographic coordinates in DEGREES as [lat, lon].
        This is target_coords_raw. Do NOT pass target_coords (the local rotated
        cell-relative encoding) -- that is not geographic and will look like noise.
    channel_names :
        Optional list of channel names, used for the title.
    channel_idx :
        Which channel to plot.
    out_dir :
        Directory for the PNGs; created if absent.
    tag :
        Extra string for the filename, e.g. "step0".
    max_points :
        Subsample above this many points. MEPS has ~800k points in-domain and
        matplotlib scatter becomes very slow well before that.
    once :
        If True, plot only the first time this (stream, channel, tag) is seen.
    """

    key = (stream_name, channel_idx, tag)
    if once and key in _PLOTTED:
        return
    _PLOTTED.add(key)

    # matplotlib is imported lazily: this is a debug path and we do not want to
    # pay the import (or require a display backend) in normal runs.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- normalise inputs to numpy, on cpu, float32 -----------------------
    t = target.detach().to(torch.float32).cpu().numpy()
    p = pred.detach().to(torch.float32).cpu().numpy()
    c = coords.detach().to(torch.float32).cpu().numpy() if torch.is_tensor(coords) else coords

    # collapse ensemble dim if present, exactly as lp_loss does
    if p.ndim == 3:
        p = p.mean(0)

    if t.shape[0] == 0:
        _logger.warning("plot_target_pred_map: %s has no points, skipping.", stream_name)
        return

    assert c.shape[0] == t.shape[0], (
        f"coords/target row mismatch for {stream_name}: {c.shape} vs {t.shape}. "
        "Are you passing target_coords_raw?"
    )
    assert p.shape[0] == t.shape[0], (
        f"pred/target row mismatch for {stream_name}: {p.shape} vs {t.shape}"
    )

    lats, lons = c[:, 0], c[:, 1]

    # --- subsample for plotting speed ------------------------------------
    n = t.shape[0]
    if n > max_points:
        sel = np.random.default_rng(0).choice(n, size=max_points, replace=False)
        lats, lons, t, p = lats[sel], lons[sel], t[sel], p[sel]
        _logger.info("plot_target_pred_map: %s subsampled %d -> %d", stream_name, n, max_points)

    tv = t[:, channel_idx]
    pv = p[:, channel_idx]
    dv = pv - tv

    ch_name = (
        channel_names[channel_idx]
        if channel_names is not None and channel_idx < len(channel_names)
        else f"ch{channel_idx}"
    )

    # --- shared colour scale for target/pred so they are comparable -------
    finite = np.isfinite(tv) & np.isfinite(pv)
    if finite.sum() == 0:
        _logger.warning("plot_target_pred_map: %s ch %s is all non-finite", stream_name, ch_name)
        return
    vmin = float(np.nanpercentile(tv[finite], 1))
    vmax = float(np.nanpercentile(tv[finite], 99))
    # symmetric scale for the difference
    dmax = float(np.nanpercentile(np.abs(dv[finite]), 99)) or 1.0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # marker size: small domains have few points and benefit from bigger markers
    s = 6.0 if len(lats) < 5_000 else 1.0

    for ax, vals, title, cmap, lo, hi in [
        (axes[0], tv, "target", "viridis", vmin, vmax),
        (axes[1], pv, "prediction", "viridis", vmin, vmax),
        (axes[2], dv, "pred - target", "RdBu_r", -dmax, dmax),
    ]:
        sc = ax.scatter(lons, lats, c=vals, s=s, cmap=cmap, vmin=lo, vmax=hi, linewidths=0)
        ax.set_title(f"{stream_name} {ch_name}: {title}")
        ax.set_xlabel("longitude [deg]")
        ax.set_ylabel("latitude [deg]")
        ax.grid(alpha=0.3, linewidth=0.5)
        fig.colorbar(sc, ax=ax, shrink=0.85)

    fname = out_dir / f"{stream_name}_{ch_name}{('_' + tag) if tag else ''}.png"
    fig.savefig(fname, dpi=110)
    plt.close(fig)

    _logger.info(
        "Wrote %s  (n=%d, lat=[%.1f, %.1f], lon=[%.1f, %.1f])",
        fname,
        len(lats),
        lats.min(),
        lats.max(),
        lons.min(),
        lons.max(),
    )


def reset() -> None:
    """Forget which (stream, channel, tag) combinations have been plotted."""
    _PLOTTED.clear()
