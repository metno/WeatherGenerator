# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Diagnostic maps of the GLOBAL LATENT SPACE, per healpix cell.

DEBUG tool. Writes PNGs and touches nothing in the training path.

What it plots, in one figure:

  row 1 : `n_components` raw latent components (arbitrary dims of ae_global_dim_embed)
  row 2 : per-cell L2 norm  +  first 3 PCA components rendered as RGB
  row 3 : the 3 PCA components individually, with their explained variance

Why the norm and PCA are the useful panels
------------------------------------------
Individual dimensions of a transformer latent are essentially arbitrary: the basis
is not privileged, so "component 0" carries no particular meaning and two runs will
put different information in it. Plotting a few raw dims is still worth doing as a
sanity check (they should not be constant, and should not be noise), but for
"is the latent spatially coherent?" the L2 norm and a PCA projection are far more
informative -- they are basis-independent.

Layout assumptions (checked against encoder.py)
-----------------------------------------------
`tokens_global` at the end of assimilate_local() is

    (rs, num_tokens_tot * ae_local_num_queries, ae_global_dim_embed)

where num_tokens_tot = num_healpix_cells + num_register_tokens + num_class_tokens,
and the register/class tokens are PREPENDED (torch.cat([tokens_global_register_class,
tokens_global], dim=1)). This module strips those leading tokens, and asserts that
what remains matches len(domain) * ae_local_num_queries. Row c of the remainder
corresponds to healpix cell domain.active_cells[c].
"""

import logging
from pathlib import Path

import numpy as np
import torch

_logger = logging.getLogger(__name__)

_PLOTTED: set = set()


def plot_latent_map(
    tokens_global: torch.Tensor,
    domain,
    components: list[int] | None = None,
    num_extra_tokens: int = 0,
    num_queries: int = 1,
    sample_idx: int = 0,
    out_dir: str | Path = "./plots/domain_check",
    tag: str = "",
    once: bool = True,
) -> None:
    """
    Plot latent components, L2 norm and PCA of the global latent, on a lat/lon map.

    Parameters
    ----------
    tokens_global :
        (rs, num_tokens * num_queries, dim) latent tensor from EncoderModule.
    domain :
        The Domain instance. Used to map compact cell index -> geographic centre.
    components :
        Which raw latent dims to plot. Default [0, 1, 2, 3].
    num_extra_tokens :
        num_register_tokens + num_class_tokens. These are PREPENDED to the cell
        tokens and are stripped here. Pass cf.num_register_tokens + cf.num_class_tokens.
    num_queries :
        cf.ae_local_num_queries. If > 1, queries are averaged per cell before plotting.
    sample_idx :
        Which element of the leading (rs) dimension to plot.
    """

    components = [0, 1, 2, 3] if components is None else components

    key = ("latent", tag, sample_idx)
    if once and key in _PLOTTED:
        return
    _PLOTTED.add(key)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- extract, strip extras, fold queries ------------------------------
    with torch.no_grad():
        t = tokens_global[sample_idx].detach().to(torch.float32).cpu().numpy()

    # strip prepended register/class tokens
    if num_extra_tokens > 0:
        t = t[num_extra_tokens * num_queries :]

    n_cells = len(domain)
    expected = n_cells * num_queries
    if t.shape[0] != expected:
        _logger.warning(
            "plot_latent_map: latent rows %d != len(domain)*num_queries %d "
            "(num_extra_tokens=%d, num_queries=%d). Skipping -- the layout assumption "
            "does not hold, plotting would be misleading.",
            t.shape[0],
            expected,
            num_extra_tokens,
            num_queries,
        )
        return

    # average over queries per cell, if any
    if num_queries > 1:
        t = t.reshape(n_cells, num_queries, -1).mean(1)

    dim = t.shape[-1]

    # --- geographic centres of the active cells ---------------------------
    lons, lats = domain.centres_lonlat()
    assert len(lons) == n_cells

    # --- L2 norm ----------------------------------------------------------
    norms = np.linalg.norm(t, axis=-1)

    # --- PCA to 3 components ---------------------------------------------
    # centre, then SVD. n_cells is small (hundreds) so this is trivial.
    tc = t - t.mean(0, keepdims=True)
    n_pca = min(3, n_cells, dim)
    try:
        _u, s, vt = np.linalg.svd(tc, full_matrices=False)
        pcs = tc @ vt[:n_pca].T
        var = (s**2) / max(len(tc) - 1, 1)
        evr = var[:n_pca] / var.sum() if var.sum() > 0 else np.zeros(n_pca)
    except np.linalg.LinAlgError:
        _logger.warning("plot_latent_map: SVD failed, skipping PCA panels.")
        pcs, evr = np.zeros((n_cells, n_pca)), np.zeros(n_pca)

    # PCA as RGB: normalise each component to [0,1] independently
    rgb = np.zeros((n_cells, 3))
    for i in range(n_pca):
        lo, hi = np.nanpercentile(pcs[:, i], [2, 98])
        rgb[:, i] = np.clip((pcs[:, i] - lo) / (hi - lo + 1e-12), 0, 1)

    # --- figure -----------------------------------------------------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ncol = max(len(components), 4)
    fig, axes = plt.subplots(3, ncol, figsize=(4.2 * ncol, 12), constrained_layout=True)
    axes = np.atleast_2d(axes)

    s_marker = 60.0 if n_cells < 500 else 12.0

    def _scatter(ax, vals, title, cmap="viridis", sym=False):
        if sym:
            m = float(np.nanpercentile(np.abs(vals), 98)) or 1.0
            lo, hi = -m, m
        else:
            lo = float(np.nanpercentile(vals, 2))
            hi = float(np.nanpercentile(vals, 98))
        sc = ax.scatter(
            lons, lats, c=vals, s=s_marker, cmap=cmap, vmin=lo, vmax=hi,
            linewidths=0.2, edgecolors="k",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("lon [deg]")
        ax.set_ylabel("lat [deg]")
        ax.grid(alpha=0.3, linewidth=0.5)
        fig.colorbar(sc, ax=ax, shrink=0.8)

    # row 1: raw components
    for j in range(ncol):
        ax = axes[0, j]
        if j < len(components):
            ci = components[j]
            if ci < dim:
                _scatter(ax, t[:, ci], f"latent dim {ci}", cmap="RdBu_r", sym=True)
            else:
                ax.set_visible(False)
        else:
            ax.set_visible(False)

    # row 2: norm, PCA-RGB, then blanks
    _scatter(axes[1, 0], norms, "per-cell L2 norm", cmap="magma")

    ax = axes[1, 1]
    ax.scatter(lons, lats, c=rgb, s=s_marker, linewidths=0.2, edgecolors="k")
    ax.set_title(f"PCA 1-3 as RGB (evr={evr[:n_pca].sum():.1%})", fontsize=10)
    ax.set_xlabel("lon [deg]")
    ax.set_ylabel("lat [deg]")
    ax.grid(alpha=0.3, linewidth=0.5)

    for j in range(2, ncol):
        axes[1, j].set_visible(False)

    # row 3: individual PCs
    for j in range(ncol):
        ax = axes[2, j]
        if j < n_pca:
            _scatter(ax, pcs[:, j], f"PC{j + 1} (evr={evr[j]:.1%})", cmap="RdBu_r", sym=True)
        else:
            ax.set_visible(False)

    fig.suptitle(
        f"Global latent space | {n_cells} cells | dim={dim}"
        + (f" | {tag}" if tag else ""),
        fontsize=13,
    )

    fname = out_dir / f"latent{('_' + tag) if tag else ''}.png"
    fig.savefig(fname, dpi=100)
    plt.close(fig)

    _logger.info(
        "Wrote %s (cells=%d dim=%d, norm=[%.2f, %.2f], PCA evr=%s)",
        fname,
        n_cells,
        dim,
        norms.min(),
        norms.max(),
        np.round(evr[:n_pca], 3).tolist(),
    )


def reset() -> None:
    """Forget what has been plotted."""
    _PLOTTED.clear()
