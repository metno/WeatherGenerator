# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import torch
import torch.nn.functional as F

stat_loss_fcts = ["stats", "kernel_crps"]  # Names of loss functions that need std computed


def gaussian(x, mu=0.0, std_dev=1.0):
    # unnormalized Gaussian where maximum is one
    return torch.exp(-0.5 * (x - mu) * (x - mu) / (std_dev * std_dev))


def normalized_gaussian(x, mu=0.0, std_dev=1.0):
    return (1 / (std_dev * np.sqrt(2.0 * np.pi))) * torch.exp(
        -0.5 * (x - mu) * (x - mu) / (std_dev * std_dev)
    )


def erf(x, mu=0.0, std_dev=1.0):
    c1 = torch.sqrt(torch.tensor(0.5 * np.pi))
    c2 = torch.sqrt(1.0 / torch.tensor(std_dev * std_dev))
    c3 = torch.sqrt(torch.tensor(2.0))
    val = c1 * (1.0 / c2 - std_dev * torch.special.erf((mu - x) / (c3 * std_dev)))
    return val


def gaussian_crps(target, ens, mu, stddev):
    # see Eq. A2 in S. Rasp and S. Lerch. Neural networks for postprocessing ensemble weather
    # forecasts. Monthly Weather Review, 146(11):3885 – 3900, 2018.
    c1 = np.sqrt(1.0 / np.pi)
    t1 = 2.0 * erf((target - mu) / stddev) - 1.0
    t2 = 2.0 * normalized_gaussian((target - mu) / stddev)
    val = stddev * ((target - mu) / stddev * t1 + t2 - c1)
    return torch.mean(val)  # + torch.mean( torch.sqrt( stddev) )


def stats(target, ens, mu, stddev):
    diff = gaussian(target, mu, stddev) - 1.0
    return torch.mean(diff * diff) + torch.mean(torch.sqrt(stddev))


def stats_normalized(target, ens, mu, stddev):
    a = normalized_gaussian(target, mu, stddev)
    max = 1 / (np.sqrt(2 * np.pi) * stddev)
    d = a - max
    return torch.mean(d * d) + torch.mean(torch.sqrt(stddev))


def stats_normalized_erf(target, ens, mu, stddev):
    delta = -torch.abs(target - mu)
    d = 0.5 + torch.special.erf(delta / (np.sqrt(2.0) * stddev))
    return torch.mean(d * d)  # + torch.mean( torch.sqrt( stddev) )


def mse_ens(target, ens, mu, stddev):
    mse_loss = torch.nn.functional.mse_loss
    return torch.stack([mse_loss(target, mem) for mem in ens], 0).mean()


def kernel_crps(
    targets,
    preds,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
    fair=True,
):
    """
    Compute kernel CRPS

    Params:
    target : shape ( num_data_points , num_channels )
    pred : shape ( ens_dim , num_data_points , num_channels)
    weights_channels : shape = (num_channels,)
    weights_points : shape = (num_data_points)

    Returns:
    loss: scalar - overall weighted CRPS
    loss_chs: [C] - per-channel CRPS (location-weighted, not channel-weighted)
    """

    ens_size = preds.shape[0]
    assert ens_size > 1, "Ensemble size has to be greater than 1 for kernel CRPS."
    assert len(preds.shape) == 3, "if data has batch dimension, remove unsqueeze() below"

    # replace NaN by 0
    mask_nan = ~torch.isnan(targets)
    targets = torch.where(mask_nan, targets, 0)
    preds = torch.where(mask_nan, preds, 0)

    # permute to enable/simply broadcasting and contractions below
    preds = preds.permute([2, 1, 0]).unsqueeze(0).to(torch.float32)
    targets = targets.permute([1, 0]).unsqueeze(0).to(torch.float32)

    mae = torch.mean(torch.abs(targets[..., None] - preds), dim=-1)

    ens_n = -1.0 / (ens_size * (ens_size - 1)) if fair else -1.0 / (ens_size**2)
    abs = torch.abs
    ens_var = torch.zeros(size=preds.shape[:-1], device=preds.device)
    # loop to reduce memory usage
    for i in range(ens_size):
        ens_var += torch.sum(ens_n * abs(preds[..., i].unsqueeze(-1) - preds[..., i + 1 :]), dim=-1)

    kcrps_locs_chs = mae + ens_var

    # apply point weighting
    if weights_points is not None:
        kcrps_locs_chs = kcrps_locs_chs * weights_points
    # apply channel weighting
    kcrps_chs = torch.mean(torch.mean(kcrps_locs_chs, 0), -1)
    if weights_channels is not None:
        kcrps_chs = kcrps_chs * weights_channels

    return torch.mean(kcrps_chs), kcrps_chs


def lp_loss(
    target: torch.Tensor,
    pred: torch.Tensor,
    p_norm: int,
    with_p_root: bool = False,
    with_mean: bool = True,
    weights_channels: torch.Tensor | None = None,
    weights_points: torch.Tensor | None = None,
):
    """
    This function computes the Lp-norm for any arbitrary integer p < inf.
    By default, the Lp-norm is normalized by the number of samples (i.e. with_mean=True).
    * For example: p=1 corresponds to MAE; p=2 corresponds to MSE.
    The samples are weighted by location if weights_points is not None.
    The norm can optionally be normalised by the pth root.
    * For example: p=2 and with_p_root=True corresponds to RMSE.
    The mean across all channels can optionally be weighted by channel weights.

    The function implements:

    loss = Mean_{channels}  ( weight_channels *
                                ( Mean_{data_pts}(|(target - pred)|**p * weights_points)
                                ) ** (1/p)
                            )

    Geometrically,

        ------------------------     -
        |                      |    |  |
        |                      |    |  |
        |                      |    |  |
        |     target - pred    | x  |wp|
        |                      |    |  |
        |                      |    |  |
        |                      |    |  |
        ------------------------     -
                    x
        ------------------------
        |          wc          |
        ------------------------

    where wp = weights_points and wc = weights_channels and "x" denotes row/col-wise multiplication.

    The computations are:
    1. weight the rows of |(target - pred)|**p by wp = weights_points (if given)
    2. take the mean over the row
    3. weight the collapsed cols by wc = weights_channels (if given)
    4. take the mean over the channel-weighted cols

    Params:
        target : tensor of shape ( num_data_points , num_channels )
        pred : tensor of shape ( ens_dim , num_data_points , num_channels)
        p_norm : integer defining the p the type of the norm
        with_mean : boolean defining whether the norm is summed or averaged
        with_p_root : boolean defining whether the p-th root of the norm is returned
        weights_channels (optional): tensor of shape = (num_channels,)
        weights_points (optional): tensor of shape = (num_data_points)

    Return:
        loss : (weighted) scalar loss (e.g. for gradient computation)
        loss_chs : losses per channel (if given with location weighting but no channel weighting)
    """

    assert type(p_norm) is int, "Only integer p supported for p-norm loss"

    mask_nan = ~torch.isnan(target)
    pred = pred[0] if pred.shape[0] == 0 else pred.mean(0)

    diff_p = torch.pow(
        torch.abs(torch.where(mask_nan, target, 0) - torch.where(mask_nan, pred, 0)), p_norm
    )
    if weights_points is not None:
        diff_p = (diff_p.transpose(1, 0) * weights_points).transpose(1, 0)
    loss_chs = diff_p.mean(0) if with_mean else diff_p.sum(0)
    loss_chs = torch.pow(loss_chs, 1.0 / p_norm) if with_p_root else loss_chs
    loss = torch.mean(loss_chs * weights_channels if weights_channels is not None else loss_chs)

    return loss, loss_chs


def mse(
    target: torch.Tensor,
    pred: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
):
    """
    Computes the mean squared error (mse).
    See lp_loss function above for a detailed explanation of arguments.
    """
    return lp_loss(
        target=target,
        pred=pred,
        p_norm=2,
        with_p_root=False,
        with_mean=True,
        weights_channels=weights_channels,
        weights_points=weights_points,
    )


def rss(
    target: torch.Tensor,
    pred: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
):
    """
    Computes the residual sum of squares (rss).
    See lp_loss function above for a detailed explanation of arguments.
    """
    return lp_loss(
        target=target,
        pred=pred,
        p_norm=2,
        with_p_root=False,
        with_mean=False,
        weights_channels=weights_channels,
        weights_points=weights_points,
    )


def rmse(
    target: torch.Tensor,
    pred: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
):
    """
    Computes the root mean squared error (rmse).
    See lp_loss function above for a detailed explanation of arguments.
    """
    return lp_loss(
        target=target,
        pred=pred,
        p_norm=2,
        with_p_root=True,
        with_mean=True,
        weights_channels=weights_channels,
        weights_points=weights_points,
    )


def mae(
    target: torch.Tensor,
    pred: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
):
    """
    Computes the mean absolute error (mae).
    See lp_loss function above for a detailed explanation of arguments.
    """
    return lp_loss(
        target=target,
        pred=pred,
        p_norm=1,
        with_p_root=False,
        with_mean=True,
        weights_channels=weights_channels,
        weights_points=weights_points,
    )


def cosine_latitude(target_coords, min_value=1e-3, max_value=1.0):
    latitudes_radian = target_coords[:, 0] * np.pi / 180
    return (max_value - min_value) * np.cos(latitudes_radian) + min_value


def gamma_decay(num_forecast_steps, gamma):
    fsteps = np.arange(num_forecast_steps)
    weights = gamma**fsteps
    return weights * (len(fsteps) / np.sum(weights))


def student_teacher_softmax(student_patches, teacher_patches, student_temp):
    """
    Cross-entropy between softmax outputs of the teacher and student networks.
    student_patches: (B, N, D) tensor
    teacher_patches: (B, N, D) tensor
    student_temp: float
    """
    loss = torch.sum(
        teacher_patches * F.log_softmax(student_patches / student_temp, dim=-1), dim=-1
    )
    loss = torch.mean(loss, dim=-1)
    return -loss.mean()


def softmax(t, s, temp):
    return torch.sum(t * F.log_softmax(s / temp, dim=-1), dim=-1)


def masked_student_teacher_patch_softmax(
    student_patches_masked,
    teacher_patches_masked,
    student_masks,
    teacher_masks,
    student_temp,
    n_masked_patches=None,
    masks_weight=None,
):
    """
    Cross-entropy between softmax outputs of the teacher and student networks.
    student_patches_masked,
    teacher_patches_masked,
    student_masks_flat,
    student_temp,
    n_masked_patches=None,
    masks_weight=None,
    """
    mask = torch.logical_and(teacher_masks, torch.logical_not(student_masks))
    loss = softmax(teacher_patches_masked[mask], student_patches_masked[mask], student_temp)
    if masks_weight is None:
        masks_weight = (
            (1 / student_masks.sum(-1).clamp(min=1.0))
            .unsqueeze(-1)
            .expand_as(student_masks)  # [student_masks_flat]
        )
    loss = loss * masks_weight[mask]
    return -loss.sum() / student_masks.shape[0]


def student_teacher_global_softmax(student_outputs, teacher_output, student_temp):
    """
    This comment is outdated TODO fix. Leaving it for now so we remember the context

    This assumes that student_outputs : list[Tensor[2*batch_size, num_class_tokens, channel_size])
                 and  teacher_outputs : Tensor[2*batch_size, num_class_tokens, channel_size]
    The 2* is because there is two global views and they are concatenated in the batch dim
    in DINOv2 as far as I can tell.
    """
    total_loss = 0
    for s in student_outputs:
        lsm = F.log_softmax(s / student_temp, dim=-1)
        loss = torch.sum(teacher_output * lsm, dim=-1)
        total_loss -= loss.mean()
    return total_loss

##############################################
#CrL
##############################################
def haar_2d(field: torch.Tensor):
    """
    One-level 2D Haar wavelet decomposition.
    field: (H, W), H and W must be even.
    Returns LL, LH, HL, HH each of shape (H//2, W//2).

    What the four subbands mean physically:
    LL: average of average. The smooth, low-frequency content of the patch. A cell with a warm temperature surrounded by warm neighbours will have a large LL coefficient. This captures the large-scale spatial mean.
    LH: average of difference (horizontal edges). Large where there is a sharp contrast between left and right columns within the smoothed rows. Captures east-west gradients.
    HL: difference of average (vertical edges). Large where there is a sharp contrast between top and bottom rows. Captures north-south gradients.
    HH: difference of difference (diagonal edges). Large at point features and diagonal patterns. Captures checkerboard-like variation.
    """
    c = 2 ** -0.5   # 1/sqrt(2) — orthonormal Haar scaling
#    Take pairs of adjacent rows. L is the row-average (low frequency — smooth variation between rows). H is the row-difference (high frequency — how much two adjacent rows differ). Both have shape (H//2, W). This is a 1D Haar along the vertical direction.
    L = (field[0::2, :] + field[1::2, :]) * c
    H = (field[0::2, :] - field[1::2, :]) * c

#    Now take pairs of adjacent columns within L and H. This is the 1D Haar along the horizontal direction, applied to both halves. All four outputs have shape (H//2, W//2).
    LL = (L[:, 0::2] + L[:, 1::2]) * c
    LH = (L[:, 0::2] - L[:, 1::2]) * c
    HL = (H[:, 0::2] + H[:, 1::2]) * c
    HH = (H[:, 0::2] - H[:, 1::2]) * c

    return LL, LH, HL, HH

def haar_wavelet_mse_local_patch(
    target: torch.Tensor,
    pred: torch.Tensor,
    local_xy: torch.Tensor,
    grid_size: int = 32,
    detail_weight: float = 2.0,
    num_levels: int = 2,
    min_points: int = 50,
):
    """
    Per-cell 2D Haar wavelet MSE using local patch coordinates.

    For each channel independently:
      - collect all points in the central cell + 1-ring
      - place them on a (grid_size x grid_size) grid using their
        local (y, z) coordinates in the central cell's rotated frame
      - apply num_levels of Haar decomposition to both target and pred grids
      - compute MSE on each subband, weighted by detail_weight for detail bands

    This function operates on the points of ONE cell patch at a time.
    It is called from _loss_wavelet_per_cell in loss_module_physical.py
    which handles the iteration over cells and aggregation.

    Args:
        target      : (num_points, num_channels) — values for this patch
        pred        : (ens_size, num_points, num_channels) — predictions
        local_xy    : (num_points, 2) — local (y, z) coords in central cell
                      frame, in radians. Range is roughly ±patch_radius.
        grid_size   : side length of the 2D grid (must be even, power of 2
                      recommended for multi-level decomposition)
        detail_weight : weight on LH, HL, HH subbands relative to LL
        num_levels  : number of Haar decomposition levels
        min_points  : minimum number of points required to compute the loss
                      (caller should already enforce this, but guard is here too)

    Returns:
        loss     : scalar wavelet MSE for this patch
        loss_chs : (num_channels,) per-channel wavelet MSE

    NOTE: Why this matters for your downscaling task. MSE in pixel space treats every point equally. 
    A prediction that is spatially smooth but offset by a constant scores the same as one that has the right mean but wrong spatial structure. Computing MSE on wavelet coefficients separates these: the LL term penalises mean errors, while LH, HL, HH penalise errors in spatial structure independently. With detail_weight=2.0 you are saying "errors in spatial gradients and edges cost twice as much as errors in the mean" — which is the right prior for downscaling where the coarse input already captures the mean and the model needs to learn fine spatial structure from the terrain and land-use geoinfo fields.
    With num_levels=2, after the first decomposition the LL subband is decomposed again, giving you a three-scale representation: the second-level LL captures very smooth large-scale variation, the second-level detail bands capture medium-scale structure, and the first-level detail bands capture fine-scale structure at the full grid resolution.
    """
    num_points, num_channels = target.shape
    dev = target.device

    if num_points < min_points:
        return (
            torch.tensor(0.0, device=dev, requires_grad=True),
            torch.zeros(num_channels, device=dev),
        )

    pred_mean = pred.mean(0)    # (num_points, num_channels)

    # --- map local_xy to grid indices ---
    # local_xy[:,0] = y coord (roughly east-west in rotated frame)
    # local_xy[:,1] = z coord (roughly north-south in rotated frame)
    y = local_xy[:, 0]
    z = local_xy[:, 1]

    # normalise to [0, grid_size) using the observed range of this patch
    # using a fixed symmetric range avoids scale inconsistency across patches
    y_range = y.abs().max().clamp(min=1e-6)
    z_range = z.abs().max().clamp(min=1e-6)

    # map [-range, +range] → [0, grid_size)
    col_idx = ((y / y_range + 1.0) * 0.5 * (grid_size - 1)).long().clamp(0, grid_size - 1)
    row_idx = ((z / z_range + 1.0) * 0.5 * (grid_size - 1)).long().clamp(0, grid_size - 1)
    bin_idx = row_idx * grid_size + col_idx     # (num_points,)
    n_bins  = grid_size * grid_size

    # --- scatter points onto grid: last-write-wins for collisions ---
    # (collisions are rare when grid_size is large enough)
    # use a count grid to compute per-bin mean when collisions occur
    target_grid = torch.zeros(n_bins, num_channels, device=dev)
    pred_grid   = torch.zeros(n_bins, num_channels, device=dev)
    count_grid  = torch.zeros(n_bins, device=dev)

    target_grid.scatter_add_(
        0, bin_idx.unsqueeze(1).expand_as(target), target.float()
    )
    pred_grid.scatter_add_(
        0, bin_idx.unsqueeze(1).expand_as(pred_mean), pred_mean.float()
    )
    count_grid.scatter_add_(0, bin_idx, torch.ones(num_points, device=dev))

    # normalise occupied bins (handles the rare collision case)
    occupied = count_grid > 0
    target_grid[occupied] = (target_grid[occupied].T / count_grid[occupied]).T
    pred_grid[occupied]   = (pred_grid[occupied].T   / count_grid[occupied]).T

    # fill empty bins with the patch mean so Haar coefficients
    # reflect actual spatial structure rather than zero-padding artefacts
    patch_mean_t = target_grid[occupied].mean(0, keepdim=True)
    patch_mean_p = pred_grid[occupied].mean(0, keepdim=True)
    target_grid[~occupied] = patch_mean_t
    pred_grid[~occupied]   = patch_mean_p

    # reshape to (grid_size, grid_size, num_channels)
    target_grid = target_grid.view(grid_size, grid_size, num_channels)
    pred_grid   = pred_grid.view(  grid_size, grid_size, num_channels)

    # ----------------------------------------------------------------
    # DEBUG: plot target_grid and pred_grid for one large patch
    # Remove after verification.
    # ----------------------------------------------------------------
    haar_wavelet_mse_local_patch._debug_plot_done = True
    if not hasattr(haar_wavelet_mse_local_patch, '_debug_plot_done') \
            and num_points > 80:
        haar_wavelet_mse_local_patch._debug_plot_done = True

        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        out_dir = "/leonardo_scratch/large/userexternal/clussana/debfigures"
        os.makedirs(out_dir, exist_ok=True)

        t_np = target_grid.detach().cpu().float().numpy()  # (grid_size, grid_size, C)
        p_np = pred_grid.detach().cpu().float().numpy()

        for c in range(num_channels):
            t_ch = t_np[:, :, c]
            p_ch = p_np[:, :, c]
            diff  = t_ch - p_ch

            # use percentile-based color scale to reveal spatial details
            # robust to outliers from empty-bin fill
            vmin = float(np.percentile(t_ch[t_ch != t_ch.mean()], 5)
                         if (t_ch != t_ch.mean()).any() else t_ch.min())
            vmax = float(np.percentile(t_ch[t_ch != t_ch.mean()], 95)
                         if (t_ch != t_ch.mean()).any() else t_ch.max())

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            im0 = axes[0].imshow(
                t_ch, origin='lower', cmap='RdBu_r',
                vmin=vmin, vmax=vmax, interpolation='nearest'
            )
            axes[0].set_title(f'target  ch={c}  n_pts={num_points}')
            axes[0].set_xlabel('col (y local)')
            axes[0].set_ylabel('row (z local)')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(
                p_ch, origin='lower', cmap='RdBu_r',
                vmin=vmin, vmax=vmax, interpolation='nearest'
            )
            axes[1].set_title(f'pred  ch={c}')
            axes[1].set_xlabel('col (y local)')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            # difference uses its own symmetric color scale
            diff_abs = np.abs(diff).max()
            diff_abs = max(diff_abs, 1e-6)
            im2 = axes[2].imshow(
                diff, origin='lower', cmap='bwr',
                vmin=-diff_abs, vmax=diff_abs, interpolation='nearest'
            )
            axes[2].set_title(f'target - pred  ch={c}')
            axes[2].set_xlabel('col (y local)')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            # overlay point positions as small dots to show coverage
            y_np = local_xy[:, 0].detach().cpu().float().numpy()
            z_np = local_xy[:, 1].detach().cpu().float().numpy()
            y_range_v = float(torch.tensor(y_np).abs().max().clamp(min=1e-6))
            z_range_v = float(torch.tensor(z_np).abs().max().clamp(min=1e-6))
            col_pts = (y_np / y_range_v + 1.0) * 0.5 * (grid_size - 1)
            row_pts = (z_np / z_range_v + 1.0) * 0.5 * (grid_size - 1)
            for ax in axes:
                ax.scatter(col_pts, row_pts,
                           s=0.5, c='black', alpha=0.3, linewidths=0)

            plt.suptitle(
                f'Wavelet patch  grid={grid_size}x{grid_size}  '
                f'n_pts={num_points}  ch={c}',
                fontsize=11
            )
            plt.tight_layout()
            out_path = os.path.join(out_dir, f'wavelet_patch_ch{c}.png')
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"[DEBUG wavelet plot] ch={c} saved to {out_path}")
    # ----------------------------------------------------------------
    # END DEBUG
    # ----------------------------------------------------------------

    # --- multi-level Haar per channel ---
    loss_chs = torch.zeros(num_channels, device=dev)
    subband_norm = 1.0 + 3.0 * detail_weight

    for c in range(num_channels):
        t_field = target_grid[:, :, c]
        p_field = pred_grid[:,   :, c]
        level_loss = torch.tensor(0.0, device=dev)

        for lvl in range(num_levels):
            t_LL, t_LH, t_HL, t_HH = haar_2d(t_field)
            p_LL, p_LH, p_HL, p_HH = haar_2d(p_field)

            scale = 1.0 / (4 ** lvl)

            level_loss = level_loss + (
                torch.mean((t_LL - p_LL) ** 2)
                + detail_weight * scale * torch.mean((t_LH - p_LH) ** 2)
                + detail_weight * scale * torch.mean((t_HL - p_HL) ** 2)
                + detail_weight * scale * torch.mean((t_HH - p_HH) ** 2)
            ) / subband_norm

            # recurse into LL subband
            t_field = t_LL
            p_field = p_LL

        loss_chs[c] = level_loss / num_levels

    loss = loss_chs.mean()
    return loss, loss_chs.detach()

def global_haar_wavelet_mse(
    target: torch.Tensor,
    pred: torch.Tensor,
    target_coords_raw: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
    grid_resolution_deg: float = 0.03,
    detail_weight: float = 2.0,
    num_levels: int = 3,
    stream_name: str = "",
):
    """
    Global 2D Haar wavelet MSE over the full field.

    Places all scattered (lat, lon) points onto a regular lat/lon grid at
    approximately the original data resolution, applies multi-level 2D Haar
    decomposition, and computes MSE between target and pred at each subband.

    Args:
        target              : (num_points, num_channels)
        pred                : (ens_size, num_points, num_channels)
        target_coords_raw   : (num_points, 2) — geographic (lat, lon) in degrees
        weights_channels    : (num_channels,) or None
        weights_points      : unused, kept for API consistency
        grid_resolution_deg : target grid spacing in degrees. Default 0.03° ≈ 3km
        detail_weight       : weight on LH, HL, HH subbands relative to LL
        num_levels          : number of Haar decomposition levels
        stream_name         : used only for debug prints

    Returns:
        loss     : scalar
        loss_chs : (num_channels,) per-channel loss
    """
    import numpy as np

    num_points, num_channels = target.shape
    dev = target.device
    pred_mean = pred.mean(0)   # (num_points, num_channels)

    lats = target_coords_raw[:, 0]   # (num_points,)
    lons = target_coords_raw[:, 1]   # (num_points,)

    lat_min = lats.min()
    lat_max = lats.max()
    lon_min = lons.min()
    lon_max = lons.max()

    # build grid dimensions on CPU (scalars only), then use on device
    n_lat_raw = max(2, int(np.ceil(((lat_max - lat_min) / grid_resolution_deg).item())) + 1)
    n_lon_raw = max(2, int(np.ceil(((lon_max - lon_min) / grid_resolution_deg).item())) + 1)

    factor = 2 ** num_levels
    n_lat = int(np.ceil(n_lat_raw / factor)) * factor
    n_lon = int(np.ceil(n_lon_raw / factor)) * factor

    # all index arithmetic stays on dev
    row_idx = ((lats - lat_min) / (lat_max - lat_min + 1e-8) * (n_lat - 1)).long().clamp(0, n_lat - 1)
    col_idx = ((lons - lon_min) / (lon_max - lon_min + 1e-8) * (n_lon - 1)).long().clamp(0, n_lon - 1)
    flat_idx = row_idx * n_lon + col_idx   # stays on dev
    n_bins = n_lat * n_lon

    # scatter onto grid — last-write-wins for collisions
    # use scatter_add + count to compute mean for colliding points
    target_grid = torch.zeros(n_bins, num_channels, device=dev)
    pred_grid   = torch.zeros(n_bins, num_channels, device=dev)
    count_grid  = torch.zeros(n_bins, device=dev)

    target_grid.scatter_add_(
        0, flat_idx.unsqueeze(1).expand_as(target), target.float()
    )
    pred_grid.scatter_add_(
        0, flat_idx.unsqueeze(1).expand_as(pred_mean), pred_mean.float()
    )
    count_grid.scatter_add_(0, flat_idx, torch.ones(num_points, device=dev))

    # normalise occupied bins
    occupied = count_grid > 0
    target_grid[occupied] = (target_grid[occupied].T / count_grid[occupied]).T
    pred_grid[occupied]   = (pred_grid[occupied].T   / count_grid[occupied]).T

    # fill empty bins with local mean of occupied bins
    field_mean_t = target_grid[occupied].mean(0, keepdim=True)
    field_mean_p = pred_grid[occupied].mean(0, keepdim=True)
    target_grid[~occupied] = field_mean_t
    pred_grid[~occupied]   = field_mean_p

    # reshape to (n_lat, n_lon, num_channels)
    target_grid = target_grid.view(n_lat, n_lon, num_channels)
    pred_grid   = pred_grid.view(  n_lat, n_lon, num_channels)


    # ----------------------------------------------------------------
    # DEBUG: plot scattered points and gridded field for global haar
    # Set DEBUG_GLOBAL_HAAR_PLOT = True to enable.
    # Remove after verification.
    # ----------------------------------------------------------------
#    DEBUG_GLOBAL_HAAR_PLOT = True
    DEBUG_GLOBAL_HAAR_PLOT = False
    DEBUG_PLOT_EVERY_N     = 1     # plot every N calls
    DEBUG_OUT_DIR          = "/leonardo_scratch/large/userexternal/clussana/debfigures/"

    _counter_key = '_global_haar_call_count'
    current_count = getattr(global_haar_wavelet_mse, _counter_key, 0)
    setattr(global_haar_wavelet_mse, _counter_key, current_count + 1)

    if DEBUG_GLOBAL_HAAR_PLOT \
    and stream_name == "NORA3" \
    and current_count % DEBUG_PLOT_EVERY_N == 0:
        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as _np

        os.makedirs(DEBUG_OUT_DIR, exist_ok=True)

        lats_np   = lats.detach().cpu().float().numpy()
        lons_np   = lons.detach().cpu().float().numpy()
        t_np      = target.detach().cpu().float().numpy()   # (num_points, C)
        p_np      = pred_mean.detach().cpu().float().numpy() # (num_points, C)
        tg_np     = target_grid.detach().cpu().float().numpy()  # (n_lat, n_lon, C)
        pg_np     = pred_grid.detach().cpu().float().numpy()

        # latitude axis for imshow (row 0 = lat_min so we flip for north-up display)
        lat_axis  = _np.linspace(float(lat_min.item()), float(lat_max.item()), n_lat)
        lon_axis  = _np.linspace(float(lon_min.item()), float(lon_max.item()), n_lon)

        for c in range(num_channels):
            t_pts  = t_np[:, c]
            p_pts  = p_np[:, c]
            t_grid = tg_np[:, :, c]
            p_grid = pg_np[:, :, c]

            # shared color scale based on target percentiles
            p2,  p98  = _np.percentile(t_pts, 2),  _np.percentile(t_pts, 98)
            d2,  d98  = _np.percentile(t_pts - p_pts, 2), _np.percentile(t_pts - p_pts, 98)
            dabs      = max(abs(d2), abs(d98), 1e-6)
            pt_size   = max(0.05, min(2.0, 80000.0 / len(lats_np)))

            fig, axes = plt.subplots(2, 3, figsize=(20, 12))

            # --- row 0: scattered points ---
            sc0 = axes[0, 0].scatter(
                lons_np, lats_np, c=t_pts,
                s=pt_size, cmap='RdBu_r', vmin=p2, vmax=p98,
                linewidths=0, rasterized=True,
            )
            axes[0, 0].set_title(f'scattered target  ch={c}  n={len(lats_np)}')
            axes[0, 0].set_xlabel('longitude'); axes[0, 0].set_ylabel('latitude')
            plt.colorbar(sc0, ax=axes[0, 0], fraction=0.046, pad=0.04)

            sc1 = axes[0, 1].scatter(
                lons_np, lats_np, c=p_pts,
                s=pt_size, cmap='RdBu_r', vmin=p2, vmax=p98,
                linewidths=0, rasterized=True,
            )
            axes[0, 1].set_title(f'scattered pred  ch={c}')
            axes[0, 1].set_xlabel('longitude')
            plt.colorbar(sc1, ax=axes[0, 1], fraction=0.046, pad=0.04)

            sc2 = axes[0, 2].scatter(
                lons_np, lats_np, c=t_pts - p_pts,
                s=pt_size, cmap='bwr', vmin=-dabs, vmax=dabs,
                linewidths=0, rasterized=True,
            )
            axes[0, 2].set_title(f'scattered target-pred  ch={c}')
            axes[0, 2].set_xlabel('longitude')
            plt.colorbar(sc2, ax=axes[0, 2], fraction=0.046, pad=0.04)

            for ax in axes[0]:
                ax.set_aspect('equal')
                ax.grid(True, lw=0.3, alpha=0.3)

            # --- row 1: gridded fields ---
            # flip vertically so north is up (row 0 = lat_min in array)
            extent = [lon_axis[0], lon_axis[-1], lat_axis[0], lat_axis[-1]]

            im0 = axes[1, 0].imshow(
                t_grid, origin='lower', extent=extent,
                cmap='RdBu_r', vmin=p2, vmax=p98,
                aspect='auto', interpolation='nearest',
            )
            axes[1, 0].set_title(f'gridded target  ch={c}  ({n_lat}x{n_lon})')
            axes[1, 0].set_xlabel('longitude'); axes[1, 0].set_ylabel('latitude')
            plt.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)

            im1 = axes[1, 1].imshow(
                p_grid, origin='lower', extent=extent,
                cmap='RdBu_r', vmin=p2, vmax=p98,
                aspect='auto', interpolation='nearest',
            )
            axes[1, 1].set_title(f'gridded pred  ch={c}')
            axes[1, 1].set_xlabel('longitude')
            plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

            t_diff_grid = t_grid - p_grid
            d2g,  d98g  = _np.percentile(t_diff_grid, 2), _np.percentile(t_diff_grid, 98)
            dabsg       = max(abs(d2g), abs(d98g), 1e-6)
            im2 = axes[1, 2].imshow(
                t_diff_grid, origin='lower', extent=extent,
                cmap='bwr', vmin=-dabsg, vmax=dabsg,
                aspect='auto', interpolation='nearest',
            )
            axes[1, 2].set_title(f'gridded target-pred  ch={c}')
            axes[1, 2].set_xlabel('longitude')
            plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)

            for ax in axes[1]:
                ax.grid(True, lw=0.3, alpha=0.3)

            plt.suptitle(
                f'global_haar  stream={stream_name}  ch={c}  '
                f'call={current_count}  '
                f'grid=({n_lat}x{n_lon})  res={grid_resolution_deg}deg  '
                f'n_pts={num_points}  '
                f'p2={p2:.3f}  p98={p98:.3f}',
                fontsize=10
            )
            plt.tight_layout()
            out_path = os.path.join(
                DEBUG_OUT_DIR,
                f'global_haar_{stream_name}_ch{c}_call{current_count:06d}.png'
            )
            plt.savefig(out_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f"[global_haar plot] {stream_name} ch={c} "
                  f"call={current_count} saved to {out_path}")
    # ----------------------------------------------------------------
    # END DEBUG
    # ----------------------------------------------------------------

    # report grid stats once
    if not getattr(global_haar_wavelet_mse, '_grid_reported', False):
        global_haar_wavelet_mse._grid_reported = True
        n_occupied = int(occupied.sum().item())
        collision_rate = max(0.0, (num_points - n_occupied) / max(num_points, 1))
        print(f"[global_haar] stream={stream_name} "
              f"grid=({n_lat},{n_lon}) "
              f"points={num_points} occupied={n_occupied} "
              f"empty={n_bins - n_occupied} "
              f"collision_rate={collision_rate:.4f}")

    # multi-level Haar per channel
    loss_chs      = torch.zeros(num_channels, device=dev)
#    subband_norm  = 1.0 + 3.0 * detail_weight
    subband_norm  = 1.0

    for c in range(num_channels):
        t_field = target_grid[:, :, c]
        p_field = pred_grid[:,   :, c]
        level_loss = torch.tensor(0.0, device=dev)

        for lvl in range(num_levels):
            t_LL, t_LH, t_HL, t_HH = haar_2d(t_field)
            p_LL, p_LH, p_HL, p_HH = haar_2d(p_field)
            # ----------------------------------------------------------------
            # DEBUG: plot Haar subbands for NORA3
            # Set DEBUG_GLOBAL_HAAR_PLOT and stream_name == "NORA3" to enable.
            # Set zoom box to None to show full field.
            # Remove after verification.
            # ----------------------------------------------------------------
            DEBUG_ZOOM_LAT_MIN  = 57.0    # south Norway
            DEBUG_ZOOM_LAT_MAX  = 62.0
            DEBUG_ZOOM_LON_MIN  = 4.0
            DEBUG_ZOOM_LON_MAX  = 12.0
            # set all four to None for full field:
            # DEBUG_ZOOM_LAT_MIN = DEBUG_ZOOM_LAT_MAX = None
            # DEBUG_ZOOM_LON_MIN = DEBUG_ZOOM_LON_MAX = None

            _haar_counter_key = '_global_haar_subband_call_count'
            _haar_count = getattr(global_haar_wavelet_mse, _haar_counter_key, 0)
            if lvl == 0:
                setattr(global_haar_wavelet_mse, _haar_counter_key, _haar_count + 1)

            if DEBUG_GLOBAL_HAAR_PLOT \
                    and stream_name == "NORA3" \
                    and _haar_count % DEBUG_PLOT_EVERY_N == 0:

                import os
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as _plt
                import numpy as _np2

                os.makedirs(DEBUG_OUT_DIR, exist_ok=True)

                # convert subbands to numpy
                t_LL_np = t_LL.detach().cpu().float().numpy()
                t_LH_np = t_LH.detach().cpu().float().numpy()
                t_HL_np = t_HL.detach().cpu().float().numpy()
                t_HH_np = t_HH.detach().cpu().float().numpy()
                p_LL_np = p_LL.detach().cpu().float().numpy()
                p_LH_np = p_LH.detach().cpu().float().numpy()
                p_HL_np = p_HL.detach().cpu().float().numpy()
                p_HH_np = p_HH.detach().cpu().float().numpy()

                subbands = {
                    'LL': (t_LL_np, p_LL_np),
                    'LH': (t_LH_np, p_LH_np),
                    'HL': (t_HL_np, p_HL_np),
                    'HH': (t_HH_np, p_HH_np),
                }

                # at level lvl the subband has been halved lvl times
                # so full-grid coordinates map to subband indices via /2^lvl
                _sf = 2 ** lvl
                _lat_ax_full = _np2.linspace(
                    float(lat_min.item()), float(lat_max.item()),
                    t_LL_np.shape[0] * 2   # LL is already halved once
                )
                _lon_ax_full = _np2.linspace(
                    float(lon_min.item()), float(lon_max.item()),
                    t_LL_np.shape[1] * 2
                )

                _do_zoom = all(v is not None for v in [
                    DEBUG_ZOOM_LAT_MIN, DEBUG_ZOOM_LAT_MAX,
                    DEBUG_ZOOM_LON_MIN, DEBUG_ZOOM_LON_MAX,
                ])

                if _do_zoom:
                    # find row/col slice in the subband array
                    # subband shape is (n_lat/2^(lvl+1), n_lon/2^(lvl+1))
                    # each subband row i corresponds to full-grid row i * 2^(lvl+1)
                    _full_n_lat = t_LL_np.shape[0] * 2   # before this Haar level
                    _full_n_lon = t_LL_np.shape[1] * 2
                    _lat_full   = _np2.linspace(float(lat_min.item()), float(lat_max.item()), _full_n_lat)
                    _lon_full   = _np2.linspace(float(lon_min.item()), float(lon_max.item()), _full_n_lon)

                    # find indices in full grid then scale to subband
                    _r0_full = int(_np2.searchsorted(_lat_full, DEBUG_ZOOM_LAT_MIN))
                    _r1_full = int(_np2.searchsorted(_lat_full, DEBUG_ZOOM_LAT_MAX)) + 1
                    _c0_full = int(_np2.searchsorted(_lon_full, DEBUG_ZOOM_LON_MIN))
                    _c1_full = int(_np2.searchsorted(_lon_full, DEBUG_ZOOM_LON_MAX)) + 1

                    # scale to subband indices (subband is 2x smaller than full grid)
                    _r0 = max(0, _r0_full // 2)
                    _r1 = min(t_LL_np.shape[0], _r1_full // 2 + 1)
                    _c0 = max(0, _c0_full // 2)
                    _c1 = min(t_LL_np.shape[1], _c1_full // 2 + 1)

                    _extent = [
                        DEBUG_ZOOM_LON_MIN, DEBUG_ZOOM_LON_MAX,
                        DEBUG_ZOOM_LAT_MIN, DEBUG_ZOOM_LAT_MAX,
                    ]
                    _xlabel = 'longitude'
                    _ylabel = 'latitude'
                    _zoom_str = (f'zoom=[{DEBUG_ZOOM_LAT_MIN},{DEBUG_ZOOM_LAT_MAX}]N '
                                 f'[{DEBUG_ZOOM_LON_MIN},{DEBUG_ZOOM_LON_MAX}]E')
                else:
                    _r0, _r1 = 0, t_LL_np.shape[0]
                    _c0, _c1 = 0, t_LL_np.shape[1]
                    _extent  = None
                    _xlabel  = 'col'
                    _ylabel  = 'row'
                    _zoom_str = 'full field'

                _imshow_kw = dict(
                    origin='lower', aspect='auto', interpolation='nearest'
                )
                if _extent is not None:
                    _imshow_kw['extent'] = _extent

                for band_name, (t_b_full, p_b_full) in subbands.items():
                    t_b    = t_b_full[_r0:_r1, _c0:_c1]
                    p_b    = p_b_full[_r0:_r1, _c0:_c1]
                    diff_b = t_b - p_b

                    if t_b.size == 0:
                        print(f"[global_haar subband] {stream_name} "
                              f"band={band_name} ch={c} lvl={lvl} "
                              f"zoom produced empty array, skipping")
                        continue

                    p2b,   p98b  = _np2.percentile(t_b,    2), _np2.percentile(t_b,    98)
                    d2b,   d98b  = _np2.percentile(diff_b, 2), _np2.percentile(diff_b, 98)
                    dabsb        = max(abs(d2b), abs(d98b), 1e-6)

                    fig, axes = _plt.subplots(1, 3, figsize=(18, 5))

                    im0 = axes[0].imshow(
                        t_b, cmap='RdBu_r', vmin=p2b, vmax=p98b, **_imshow_kw
                    )
                    axes[0].set_title(f'target {band_name}  ch={c}  lvl={lvl}')
                    axes[0].set_xlabel(_xlabel)
                    axes[0].set_ylabel(_ylabel)
                    _plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

                    im1 = axes[1].imshow(
                        p_b, cmap='RdBu_r', vmin=p2b, vmax=p98b, **_imshow_kw
                    )
                    axes[1].set_title(f'pred {band_name}  ch={c}  lvl={lvl}')
                    axes[1].set_xlabel(_xlabel)
                    _plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                    im2 = axes[2].imshow(
                        diff_b, cmap='bwr', vmin=-dabsb, vmax=dabsb, **_imshow_kw
                    )
                    axes[2].set_title(f'target-pred {band_name}  ch={c}  lvl={lvl}')
                    axes[2].set_xlabel(_xlabel)
                    _plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

                    for ax in axes:
                        ax.grid(True, lw=0.3, alpha=0.3)

                    _plt.suptitle(
                        f'global_haar subbands  stream={stream_name}  '
                        f'band={band_name}  ch={c}  lvl={lvl}  '
                        f'call={_haar_count}  '
                        f'subband shape={t_b.shape}  '
                        f'{_zoom_str}  '
                        f'p2={p2b:.3f}  p98={p98b:.3f}',
                        fontsize=10
                    )
                    _plt.tight_layout()
                    out_path = os.path.join(
                        DEBUG_OUT_DIR,
                        f'global_haar_{stream_name}_subbands'
                        f'_ch{c}_lvl{lvl}_{band_name}'
                        f'_call{_haar_count:06d}.png'
                    )
                    _plt.savefig(out_path, dpi=120, bbox_inches='tight')
                    _plt.close(fig)
                    print(f"[global_haar subband] {stream_name} "
                          f"band={band_name} ch={c} lvl={lvl} "
                          f"shape={t_b.shape} "
                          f"call={_haar_count} saved to {out_path}")
            # ----------------------------------------------------------------
            # END DEBUG subbands
            # ----------------------------------------------------------------            
            scale = 1.0 / (4 ** lvl)


#            level_loss = level_loss + (
#                torch.mean((t_LL - p_LL) ** 2)
#                + detail_weight * scale * torch.mean((t_LH - p_LH) ** 2)
#                + detail_weight * scale * torch.mean((t_HL - p_HL) ** 2)
#                + detail_weight * scale * torch.mean((t_HH - p_HH) ** 2)
#            ) / subband_norm
            # mask out cells where all detail subbands are identical
            # (these are empty grid cells filled with field mean)
            _active_mask = (
                (t_LH != p_LH) | (t_HL != p_HL) | (t_HH != p_HH)
            )
            _n_active = _active_mask.sum().clamp(min=1)

            if _n_active > 0:
                level_loss = level_loss + (
                    scale * ((t_LH - p_LH) ** 2)[_active_mask].mean()
                    + scale * ((t_HL - p_HL) ** 2)[_active_mask].mean()
                    + scale * ((t_HH - p_HH) ** 2)[_active_mask].mean()
                ) / 3.0
                # ----------------------------------------------------------------
                # DEBUG: print subband loss components
                # ----------------------------------------------------------------
                if stream_name == "NORA3":
                    _lh_loss = ((t_LH - p_LH) ** 2)[_active_mask].mean().item()
                    _hl_loss = ((t_HL - p_HL) ** 2)[_active_mask].mean().item()
                    _hh_loss = ((t_HH - p_HH) ** 2)[_active_mask].mean().item()
                    print(
                        f"[haar components] stream={stream_name} ch={c} lvl={lvl} "
                        f"scale={scale:.6f}  "
                        f"active={_n_active.item()}/{t_LH.numel()}  "
                        f"LH={_lh_loss:.6f}  "
                        f"HL={_hl_loss:.6f}  "
                        f"HH={_hh_loss:.6f}  "
                        f"scaled_LH={scale * _lh_loss:.6f}  "
                        f"scaled_HL={scale * _hl_loss:.6f}  "
                        f"scaled_HH={scale * _hh_loss:.6f}  "
                        f"level_contrib="
                        f"{(scale*_lh_loss + scale*_hl_loss + scale*_hh_loss) / 3.0:.6f}"
                    )
                # ----------------------------------------------------------------
                # END DEBUG components
                # ----------------------------------------------------------------

            t_field = t_LL
            p_field = p_LL

        loss_chs[c] = level_loss / num_levels

    if weights_channels is not None:
        loss = torch.mean(loss_chs * weights_channels.to(dev))
    else:
        loss = torch.mean(loss_chs)

    return loss, loss_chs
