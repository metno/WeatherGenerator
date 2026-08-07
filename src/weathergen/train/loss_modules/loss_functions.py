# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math

import numpy as np
import torch
import torch.nn.functional as F

stat_loss_fcts = ["stats", "kernel_crps"]  # Names of loss functions that need std computed

# ---------------------------------------------------------------------------
# Debug-plot configuration for the *_reshape_varweighted Haar losses.
# Central switchboard so the scattered per-function DEBUG_* constants don't
# each need editing. Set HAAR_DEBUG_ENABLE=True to turn plots on. Plots are
# produced for any stream whose name is in HAAR_DEBUG_STREAMS.
# ---------------------------------------------------------------------------
HAAR_DEBUG_ENABLE = False
HAAR_DEBUG_STREAMS = {"MEPS", "ERA5"}
HAAR_DEBUG_OUT_DIR = "/capstor/scratch/cscs/clussana/debug_run/"
HAAR_DEBUG_EVERY_N = 10  # plot every Nth counted call (per stream)


def _haar_debug_on(stream_name: str) -> bool:
    """True if debug plotting should run for this stream."""
    return HAAR_DEBUG_ENABLE and stream_name in HAAR_DEBUG_STREAMS


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

def crps_kernel_pointwise(
    target: torch.Tensor,
    preds: torch.Tensor,
    fair: bool = True,
) -> torch.Tensor:
    """
    Per-point ensemble kernel CRPS (no weighting, no reduction).

        CRPS(x_1..x_E ; y) = (1/E) sum_i |x_i - y|
                             - c_E * sum_{i<j} |x_i - x_j|

    with c_E = 1/(E(E-1)) for the fair estimator (unbiased for finite
    ensembles, Ferro 2014) or 1/E^2 for the classical one. This matches the
    pair-summation convention of the pre-existing kernel_crps implementation.

    Params:
        target : tensor of arbitrary shape T
        preds  : tensor of shape (E, *T) — ensemble dim first
        fair   : use the fair (unbiased) spread normalization

    Returns:
        per-point CRPS of shape T (same dtype/device as inputs after upcast
        by the caller; this function does not change dtype)
    """
    ens_size = preds.shape[0]
    assert ens_size > 1, "Ensemble size has to be greater than 1 for kernel CRPS."

    # skill term: mean_i |x_i - y|
    skill = torch.mean(torch.abs(preds - target.unsqueeze(0)), dim=0)

    # spread term: sum over unordered member pairs, looped to bound memory
    c_e = 1.0 / (ens_size * (ens_size - 1)) if fair else 1.0 / (ens_size**2)
    spread = torch.zeros_like(skill)
    for i in range(ens_size - 1):
        spread = spread + torch.sum(
            torch.abs(preds[i].unsqueeze(0) - preds[i + 1 :]), dim=0
        )

    return skill - c_e * spread

def kernel_crps(
    targets,
    preds,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
    fair: bool = True,
):
    """
    Compute kernel CRPS in physical space.

    Params:
        targets          : ( num_data_points , num_channels )
        preds            : ( ens_size , num_data_points , num_channels )
        weights_channels : ( num_channels, ) or None
        weights_points   : ( num_data_points, ) or None
        fair             : fair (unbiased) spread normalization

    Returns:
        loss     : scalar — overall weighted CRPS
        loss_chs : (C,) per-channel CRPS (location-weighted, not channel-weighted)
    """
    ens_size = preds.shape[0]
    assert ens_size > 1, "Ensemble size has to be greater than 1 for kernel CRPS."
    assert preds.dim() == 3, "if data has batch dimension, adapt kernel_crps"

    # replace NaN (in target) by 0 in both tensors: NaN points contribute 0
    mask_nan = ~torch.isnan(targets)
    targets = torch.where(mask_nan, targets, 0)
    preds = torch.where(mask_nan, preds, 0)  # (N,C) mask broadcasts over (E,N,C)

    kcrps_pts_chs = crps_kernel_pointwise(
        targets.to(torch.float32), preds.to(torch.float32), fair=fair
    )  # (N, C)

    # apply point weighting
    if weights_points is not None:
        kcrps_pts_chs = kcrps_pts_chs * weights_points.unsqueeze(-1)

    # per-channel CRPS, then channel weighting
    kcrps_chs = kcrps_pts_chs.mean(0)
    if weights_channels is not None:
        kcrps_chs = kcrps_chs * weights_channels

    return torch.mean(kcrps_chs), kcrps_chs

def kernel_crps_OLD(
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
    # TEMP diagnostic
#    print(f"target={tuple(target.shape)} pred={tuple(pred.shape)} "
#          f"nan_t={torch.isnan(target).sum().item()} loss_chs={loss_chs}")
#    print(f"nan_p={torch.isnan(pred).sum().item()} "
#          f"pmin={pred.min().item():.3e} pmax={pred.max().item():.3e}")
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
    return (max_value - min_value) * torch.cos(latitudes_radian) + min_value


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
def haar_2d_OLD(field: torch.Tensor):
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

def haar_2d(field: torch.Tensor):
    """
    One-level 2D Haar wavelet decomposition over the LAST TWO dims.
    field: (..., H, W), H and W must be even.
    Returns LL, LH, HL, HH each of shape (..., H//2, W//2).
    """
    c = 2 ** -0.5  # 1/sqrt(2) — orthonormal Haar scaling

    # 1D Haar along rows (second-to-last dim)
    L = (field[..., 0::2, :] + field[..., 1::2, :]) * c
    H = (field[..., 0::2, :] - field[..., 1::2, :]) * c

    # 1D Haar along columns (last dim)
    LL = (L[..., 0::2] + L[..., 1::2]) * c
    LH = (L[..., 0::2] - L[..., 1::2]) * c
    HL = (H[..., 0::2] + H[..., 1::2]) * c
    HH = (H[..., 0::2] - H[..., 1::2]) * c

    return LL, LH, HL, HH

def haar_wavelet_mse_local_patch(
    target: torch.Tensor,
    pred: torch.Tensor,
    local_xy: torch.Tensor,
    grid_size: int = 32,
    detail_weight: float = 2.0,
    num_levels: int = 2,
    min_points: int = 50,
    stream_name: str = "",
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
#        print(f"[global_haar] stream={stream_name} "
#              f"grid=({n_lat},{n_lon}) "
#              f"points={num_points} occupied={n_occupied} "
#              f"empty={n_bins - n_occupied} "
#              f"collision_rate={collision_rate:.4f}")

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
                if DEBUG_GLOBAL_HAAR_PLOT \
                    and stream_name == "NORA3" \
                    and _haar_count % DEBUG_PLOT_EVERY_N == 0:
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

    # normalise by target field variance to make loss scale-invariant
    # this brings wavelet loss onto the same scale as MSE regardless of
    # the field magnitude
    field_var = target.float().var().clamp(min=1e-8)
    loss     = loss / field_var
    loss_chs = loss_chs / field_var

    return loss, loss_chs

###########################################################################
###########################################################################
###########################################################################
###########################################################################
def global_haar_wavelet_reshape(
    target: torch.Tensor,
    pred: torch.Tensor,
    target_coords_raw: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
    template_path: str = "",
    detail_weight: float = 2.0,
    num_levels: int = 3,
    stream_name: str = "",
):
    """
    Global 2D Haar wavelet MSE using template-based regridding.

    Follows MetnoParser.regrid() exactly:
      1. Load template latitude/longitude (2D, shape (ny, nx))
      2. For each output grid point find the nearest input point (NearestNDInterpolator)
      3. Gather: values_on_grid = values[Isort], reshape to (ny, nx)
      4. Apply multi-level Haar decomposition and compute MSE on detail subbands

    Template output grid is cached. Isort is recomputed each call because
    input points change batch to batch due to masking.
    """
    import math as _math_rs
    import numpy as _np_rs

    num_points, num_channels = target.shape
    dev       = target.device
    pred_mean = pred.mean(0)

    # --- load and cache template output grid ---
    _grid_key  = f'_template_grid_{template_path}'
    _shape_key = f'_template_shape_{template_path}'

    template_grid = getattr(global_haar_wavelet_reshape, _grid_key,  None)
    ny_nx         = getattr(global_haar_wavelet_reshape, _shape_key, None)

    if template_grid is None:
        if not template_path:
            _key = f'_no_template_reported_{stream_name}'
            if not getattr(global_haar_wavelet_reshape, _key, False):
                setattr(global_haar_wavelet_reshape, _key, True)
                print(
                    f"[global_haar_reshape] stream={stream_name} "
                    f"template_path is empty. Setting loss to zero."
                )
            return (
                torch.tensor(0.0, device=dev, requires_grad=True),
                torch.zeros(num_channels, device=dev),
            )

        try:
            import xarray as _xr_rs

            template = _xr_rs.open_dataset(template_path)

            # latitude/longitude are 2D arrays with dims (y, x)
            # flatten in C-order (row-major) — same as metno_parser
            olat = template.latitude.values.flatten()   # (ny*nx,)
            olon = template.longitude.values.flatten()

            # ny and nx from the coordinate axes — same as metno_parser:
            # len(y) rows, len(x) columns
            ny = len(template.y.values)
            nx = len(template.x.values)

            # cache output grid coordinates (fixed for all batches)
            template_grid = _np_rs.stack([olat, olon], axis=1)  # (ny*nx, 2)
            setattr(global_haar_wavelet_reshape, _grid_key,  template_grid)
            setattr(global_haar_wavelet_reshape, _shape_key, (ny, nx))
            ny_nx = (ny, nx)

            print(
                f"[global_haar_reshape] stream={stream_name} "
                f"template loaded: grid=({ny}x{nx})  "
                f"n_output_points={ny*nx}"
            )

        except Exception as _e:
            _key = f'_template_error_reported_{stream_name}'
            if not getattr(global_haar_wavelet_reshape, _key, False):
                setattr(global_haar_wavelet_reshape, _key, True)
                print(
                    f"[global_haar_reshape] stream={stream_name} "
                    f"failed to load template '{template_path}': {_e}. "
                    f"Setting loss to zero."
                )
            return (
                torch.tensor(0.0, device=dev, requires_grad=True),
                torch.zeros(num_channels, device=dev),
            )

    ny, nx = ny_nx

    # --- build Isort fresh each call ---
    # identical to MetnoParser.get_sorting:
    #   ipoints = input lat/lon pairs
    #   opoints = output grid lat/lon pairs (from template)
    #   Isort[i] = index of nearest input point to output position i
    import scipy.interpolate as _sci_rs

    ilat    = target_coords_raw[:, 0].cpu().numpy()
    ilon    = target_coords_raw[:, 1].cpu().numpy()
    ipoints = _np_rs.concatenate([ilat[:, None], ilon[:, None]], axis=1)

    interpolator = _sci_rs.NearestNDInterpolator(
        ipoints, _np_rs.arange(len(ilat))
    )
    Isort = interpolator(template_grid).astype(int)  # (ny*nx,)

    # --- gather values onto grid ---
    # identical to MetnoParser: all_values[:, Isort, i, :]
    # then np.reshape(values, [time, ny, nx])
    Isort_t = torch.from_numpy(Isort).long().to(dev)
    t_flat  = target.float()[Isort_t]      # (ny*nx, C)
    p_flat  = pred_mean.float()[Isort_t]

    # reshape to (ny, nx, C) — C-order matches the flatten order above
    t_grid_raw = t_flat.view(ny, nx, num_channels)
    p_grid_raw = p_flat.view(ny, nx, num_channels)

    # pad to next multiple of 2^num_levels for Haar compatibility
    # pad rows and columns independently to avoid mixing spatial dimensions
    factor = 2 ** num_levels
    ny_p   = int(_math_rs.ceil(ny / factor)) * factor
    nx_p   = int(_math_rs.ceil(nx / factor)) * factor

    if ny_p > ny or nx_p > nx:
        t_grid = torch.zeros(ny_p, nx_p, num_channels, device=dev)
        p_grid = torch.zeros(ny_p, nx_p, num_channels, device=dev)
        t_grid[:ny, :nx, :] = t_grid_raw
        p_grid[:ny, :nx, :] = p_grid_raw
    else:
        t_grid = t_grid_raw
        p_grid = p_grid_raw

    # --- debug parameters ---
    DEBUG_RESHAPE_PLOT = True
    DEBUG_PLOT_EVERY_N = 100
    DEBUG_OUT_DIR      = "/leonardo_scratch/large/userexternal/clussana/debfigures/"
#    DEBUG_ZOOM_LAT_MIN = 57.0
#    DEBUG_ZOOM_LAT_MAX = 62.0
#    DEBUG_ZOOM_LON_MIN = 4.0
#    DEBUG_ZOOM_LON_MAX = 12.0
    # set all four to None for full field:
    DEBUG_ZOOM_LAT_MIN = DEBUG_ZOOM_LAT_MAX = DEBUG_ZOOM_LON_MIN = DEBUG_ZOOM_LON_MAX = None

    _counter_key = '_global_haar_reshape_call_count'
    _call_count  = getattr(global_haar_wavelet_reshape, _counter_key, 0)
    if stream_name == "NORA3":
        setattr(global_haar_wavelet_reshape, _counter_key, _call_count + 1)

    lat_min_v = float(template_grid[:, 0].min())
    lat_max_v = float(template_grid[:, 0].max())
    lon_min_v = float(template_grid[:, 1].min())
    lon_max_v = float(template_grid[:, 1].max())

    # --- plot full gridded field ---
    if DEBUG_RESHAPE_PLOT \
            and stream_name == "NORA3" \
            and _call_count % DEBUG_PLOT_EVERY_N == 0:

        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as _plt_rs

        os.makedirs(DEBUG_OUT_DIR, exist_ok=True)

        _do_zoom = all(v is not None for v in [
            DEBUG_ZOOM_LAT_MIN, DEBUG_ZOOM_LAT_MAX,
            DEBUG_ZOOM_LON_MIN, DEBUG_ZOOM_LON_MAX,
        ])

        for c in range(num_channels):
            t_np    = t_grid[:ny, :nx, c].detach().cpu().float().numpy()
            p_np    = p_grid[:ny, :nx, c].detach().cpu().float().numpy()
            diff_np = t_np - p_np

            if _do_zoom:
                _lat_ax = _np_rs.linspace(lat_min_v, lat_max_v, ny)
                _lon_ax = _np_rs.linspace(lon_min_v, lon_max_v, nx)
                _r0 = max(0, int(_np_rs.searchsorted(_lat_ax, DEBUG_ZOOM_LAT_MIN)))
                _r1 = min(ny, int(_np_rs.searchsorted(_lat_ax, DEBUG_ZOOM_LAT_MAX)) + 1)
                _c0 = max(0, int(_np_rs.searchsorted(_lon_ax, DEBUG_ZOOM_LON_MIN)))
                _c1 = min(nx, int(_np_rs.searchsorted(_lon_ax, DEBUG_ZOOM_LON_MAX)) + 1)
                t_plot    = t_np[_r0:_r1, _c0:_c1]
                p_plot    = p_np[_r0:_r1, _c0:_c1]
                diff_plot = diff_np[_r0:_r1, _c0:_c1]
                _extent   = [DEBUG_ZOOM_LON_MIN, DEBUG_ZOOM_LON_MAX,
                             DEBUG_ZOOM_LAT_MIN, DEBUG_ZOOM_LAT_MAX]
                _zoom_str = (f'zoom=[{DEBUG_ZOOM_LAT_MIN},{DEBUG_ZOOM_LAT_MAX}]N '
                             f'[{DEBUG_ZOOM_LON_MIN},{DEBUG_ZOOM_LON_MAX}]E')
            else:
                t_plot    = t_np
                p_plot    = p_np
                diff_plot = diff_np
                _extent   = [lon_min_v, lon_max_v, lat_min_v, lat_max_v]
                _zoom_str = 'full field'

            if t_plot.size == 0:
                continue

            p2,  p98 = _np_rs.percentile(t_plot,    2), _np_rs.percentile(t_plot,   98)
            d2,  d98 = _np_rs.percentile(diff_plot, 2), _np_rs.percentile(diff_plot, 98)
            dabs     = max(abs(d2), abs(d98), 1e-6)

            _imshow_kw = dict(origin='lower', aspect='auto',
                              interpolation='nearest', extent=_extent)

            fig, axes = _plt_rs.subplots(1, 3, figsize=(18, 5))

            im0 = axes[0].imshow(t_plot, cmap='RdBu_r',
                                 vmin=p2, vmax=p98, **_imshow_kw)
            axes[0].set_title(f'target grid  ch={c}  ({ny}x{nx})')
            axes[0].set_xlabel('longitude')
            axes[0].set_ylabel('latitude')
            _plt_rs.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(p_plot, cmap='RdBu_r',
                                 vmin=p2, vmax=p98, **_imshow_kw)
            axes[1].set_title(f'pred grid  ch={c}')
            axes[1].set_xlabel('longitude')
            _plt_rs.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            im2 = axes[2].imshow(diff_plot, cmap='bwr',
                                 vmin=-dabs, vmax=dabs, **_imshow_kw)
            axes[2].set_title(f'target-pred  ch={c}')
            axes[2].set_xlabel('longitude')
            _plt_rs.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            for ax in axes:
                ax.grid(True, lw=0.3, alpha=0.3)

            _plt_rs.suptitle(
                f'global_haar_reshape  stream={stream_name}  ch={c}  '
                f'call={_call_count}  grid=({ny}x{nx})  '
                f'{_zoom_str}  p2={p2:.3f}  p98={p98:.3f}',
                fontsize=10
            )
            _plt_rs.tight_layout()
            out_path = os.path.join(
                DEBUG_OUT_DIR,
                f'haar_reshape_{stream_name}_grid_ch{c}_call{_call_count:06d}.png'
            )
            _plt_rs.savefig(out_path, dpi=120, bbox_inches='tight')
            _plt_rs.close(fig)
            print(f"[haar_reshape grid] {stream_name} ch={c} "
                  f"call={_call_count} saved to {out_path}")

    # --- multi-level Haar per channel ---
    loss_chs = torch.zeros(num_channels, device=dev)

    for c in range(num_channels):
        t_field    = t_grid[:, :, c]
        p_field    = p_grid[:, :, c]
        level_loss = torch.tensor(0.0, device=dev)

        for lvl in range(num_levels):
            t_LL, t_LH, t_HL, t_HH = haar_2d(t_field)
            p_LL, p_LH, p_HL, p_HH = haar_2d(p_field)

            # ----------------------------------------------------------------
            # DEBUG: plot subbands for NORA3
            # ----------------------------------------------------------------
            if DEBUG_RESHAPE_PLOT \
                    and stream_name == "NORA3" \
                    and _call_count % DEBUG_PLOT_EVERY_N == 0:

                import os
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as _plt_sb

                os.makedirs(DEBUG_OUT_DIR, exist_ok=True)

                subbands = {
                    'LL': (t_LL, p_LL),
                    'LH': (t_LH, p_LH),
                    'HL': (t_HL, p_HL),
                    'HH': (t_HH, p_HH),
                }

                _do_zoom_sb = all(v is not None for v in [
                    DEBUG_ZOOM_LAT_MIN, DEBUG_ZOOM_LAT_MAX,
                    DEBUG_ZOOM_LON_MIN, DEBUG_ZOOM_LON_MAX,
                ])

                _full_nr = t_LL.shape[0] * 2
                _full_nc = t_LL.shape[1] * 2

                if _do_zoom_sb:
                    _lat_sb = _np_rs.linspace(lat_min_v, lat_max_v, _full_nr)
                    _lon_sb = _np_rs.linspace(lon_min_v, lon_max_v, _full_nc)
                    _r0s = max(0,
                               int(_np_rs.searchsorted(_lat_sb, DEBUG_ZOOM_LAT_MIN)) // 2)
                    _r1s = min(t_LL.shape[0],
                               int(_np_rs.searchsorted(_lat_sb, DEBUG_ZOOM_LAT_MAX)) // 2 + 1)
                    _c0s = max(0,
                               int(_np_rs.searchsorted(_lon_sb, DEBUG_ZOOM_LON_MIN)) // 2)
                    _c1s = min(t_LL.shape[1],
                               int(_np_rs.searchsorted(_lon_sb, DEBUG_ZOOM_LON_MAX)) // 2 + 1)
                    _extent_sb   = [DEBUG_ZOOM_LON_MIN, DEBUG_ZOOM_LON_MAX,
                                    DEBUG_ZOOM_LAT_MIN, DEBUG_ZOOM_LAT_MAX]
                    _zoom_str_sb = (f'zoom=[{DEBUG_ZOOM_LAT_MIN},{DEBUG_ZOOM_LAT_MAX}]N '
                                    f'[{DEBUG_ZOOM_LON_MIN},{DEBUG_ZOOM_LON_MAX}]E')
                    _xlabel = 'longitude'
                    _ylabel = 'latitude'
                else:
                    _r0s, _r1s = 0, t_LL.shape[0]
                    _c0s, _c1s = 0, t_LL.shape[1]
                    _extent_sb   = None
                    _zoom_str_sb = 'full field'
                    _xlabel = 'col'
                    _ylabel = 'row'

                _imshow_sb = dict(origin='lower', aspect='auto',
                                  interpolation='nearest')
                if _extent_sb is not None:
                    _imshow_sb['extent'] = _extent_sb

                for band_name, (t_b, p_b) in subbands.items():
                    t_b_np = t_b[_r0s:_r1s, _c0s:_c1s].detach().cpu().float().numpy()
                    p_b_np = p_b[_r0s:_r1s, _c0s:_c1s].detach().cpu().float().numpy()
                    diff_b = t_b_np - p_b_np

                    if t_b_np.size == 0:
                        continue

                    p2b,  p98b = _np_rs.percentile(t_b_np, 2),  _np_rs.percentile(t_b_np, 98)
                    d2b,  d98b = _np_rs.percentile(diff_b,  2),  _np_rs.percentile(diff_b,  98)
                    dabsb      = max(abs(d2b), abs(d98b), 1e-6)

                    fig, axes = _plt_sb.subplots(1, 3, figsize=(18, 5))

                    im0 = axes[0].imshow(t_b_np, cmap='RdBu_r',
                                         vmin=p2b, vmax=p98b, **_imshow_sb)
                    axes[0].set_title(f'target {band_name}  ch={c}  lvl={lvl}')
                    axes[0].set_xlabel(_xlabel)
                    axes[0].set_ylabel(_ylabel)
                    _plt_sb.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

                    im1 = axes[1].imshow(p_b_np, cmap='RdBu_r',
                                         vmin=p2b, vmax=p98b, **_imshow_sb)
                    axes[1].set_title(f'pred {band_name}  ch={c}  lvl={lvl}')
                    axes[1].set_xlabel(_xlabel)
                    _plt_sb.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                    im2 = axes[2].imshow(diff_b, cmap='bwr',
                                         vmin=-dabsb, vmax=dabsb, **_imshow_sb)
                    axes[2].set_title(f'target-pred {band_name}  ch={c}  lvl={lvl}')
                    axes[2].set_xlabel(_xlabel)
                    _plt_sb.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

                    for ax in axes:
                        ax.grid(True, lw=0.3, alpha=0.3)

                    _plt_sb.suptitle(
                        f'haar_reshape subbands  stream={stream_name}  '
                        f'band={band_name}  ch={c}  lvl={lvl}  '
                        f'call={_call_count}  shape={t_b_np.shape}  '
                        f'{_zoom_str_sb}  p2={p2b:.3f}  p98={p98b:.3f}',
                        fontsize=10
                    )
                    _plt_sb.tight_layout()
                    out_path = os.path.join(
                        DEBUG_OUT_DIR,
                        f'haar_reshape_{stream_name}_subbands'
                        f'_ch{c}_lvl{lvl}_{band_name}'
                        f'_call{_call_count:06d}.png'
                    )
                    _plt_sb.savefig(out_path, dpi=120, bbox_inches='tight')
                    _plt_sb.close(fig)
                    print(f"[haar_reshape subband] {stream_name} "
                          f"band={band_name} ch={c} lvl={lvl} "
                          f"shape={t_b_np.shape} "
                          f"call={_call_count} saved to {out_path}")
            # ----------------------------------------------------------------
            # END DEBUG subbands
            # ----------------------------------------------------------------

            #_active_mask = (
            #    (t_LH != p_LH) | (t_HL != p_HL) | (t_HH != p_HH)
            #)
            # to remove active_mask
            _active_mask = torch.ones_like(t_LH, dtype=torch.bool)
            _n_active = _active_mask.sum().clamp(min=1)

            if _n_active > 0:
#                scale = 1.0 / (4 ** lvl)
                scale = 1.0
                level_loss = level_loss + (
                    scale * ((t_LH - p_LH) ** 2)[_active_mask].mean()
                    + scale * ((t_HL - p_HL) ** 2)[_active_mask].mean()
                    + scale * ((t_HH - p_HH) ** 2)[_active_mask].mean()
                ) / 3.0

            t_field = t_LL
            p_field = p_LL

            field_var    = t_grid[:, :, c].var().clamp(min=1e-8)
            loss_chs[c] = level_loss / (num_levels * field_var)

        loss_chs[c] = level_loss / num_levels

    if weights_channels is not None:
        loss = torch.mean(loss_chs * weights_channels.to(dev))
    else:
        loss = torch.mean(loss_chs)

    return loss, loss_chs
###########################################################################
###########################################################################
###########################################################################
###########################################################################


def healpix_cell_mse(
    target: torch.Tensor,
    pred: torch.Tensor,
    target_coords_lens: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
    stream_name: str = "",
):
    """
    MSE between per-HEALPix-cell averages of target and pred.

    For each cell that has at least one point, averages the target values
    and pred values independently, then computes MSE between those averages.
    Cells with zero points are excluded from the loss.

    Args:
        target              : (num_points, num_channels)
        pred                : (ens_size, num_points, num_channels)
        target_coords_lens  : (num_cells,) int32 — points per cell
        weights_channels    : (num_channels,) or None
        weights_points      : unused, kept for API consistency

    Returns:
        loss     : scalar
        loss_chs : (num_channels,) per-channel loss
    """
    num_points, num_channels = target.shape
    dev = target.device
    pred_mean = pred.mean(0)   # (num_points, num_channels)

    num_cells = target_coords_lens.shape[0]
    tc_lens = target_coords_lens.long().to(dev)

    # build cumulative offsets into the flat tensors
    cumlen = torch.cat([
        torch.zeros(1, dtype=torch.long, device=dev),
        tc_lens.cumsum(0)
    ])

    # accumulate per-cell sums using scatter_add
    cell_ids = torch.repeat_interleave(
        torch.arange(num_cells, device=dev), tc_lens
    )   # (num_points,) — cell id for each point

    target_cell_sum = torch.zeros(num_cells, num_channels, device=dev)
    pred_cell_sum   = torch.zeros(num_cells, num_channels, device=dev)
    count           = torch.zeros(num_cells, device=dev)

    target_cell_sum.scatter_add_(
        0, cell_ids.unsqueeze(1).expand_as(target), target.float()
    )
    pred_cell_sum.scatter_add_(
        0, cell_ids.unsqueeze(1).expand_as(pred_mean), pred_mean.float()
    )
    count.scatter_add_(0, cell_ids, torch.ones(num_points, device=dev))

    # only use cells that have at least one point
    occupied = count > 0
    n_occupied = occupied.sum().clamp(min=1)

    target_avg = target_cell_sum[occupied] / count[occupied].unsqueeze(1)
    pred_avg   = pred_cell_sum[occupied]   / count[occupied].unsqueeze(1)

    # ----------------------------------------------------------------
    # DEBUG: plot per-cell averages of target and pred for NORA3 only
    # Set DEBUG_CELL_MSE_PLOT = True to enable.
    # Remove after verification.
    # ----------------------------------------------------------------
    DEBUG_CELL_MSE_PLOT = False
    DEBUG_PLOT_EVERY_N  = 1
    DEBUG_OUT_DIR       = "/leonardo_scratch/large/userexternal/clussana/debfigures/"
    DEBUG_STREAM_NAME   = "NORA3"

    _counter_key = '_healpix_cell_mse_call_count'
    _call_count  = getattr(healpix_cell_mse, _counter_key, 0)
    setattr(healpix_cell_mse, _counter_key, _call_count + 1)

    if DEBUG_CELL_MSE_PLOT \
            and stream_name == DEBUG_STREAM_NAME \
            and _call_count % DEBUG_PLOT_EVERY_N == 0:

        import os
        import math as _math3
        import numpy as _np3
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as _plt3
        from astropy_healpix.healpy import pix2ang as _pix2ang3

        os.makedirs(DEBUG_OUT_DIR, exist_ok=True)

        # recover occupied cell ids
        occupied_cell_ids = torch.where(occupied)[0].cpu().numpy()

        # infer healpix level from num_cells
        hl  = round(_math3.log(num_cells / 12, 4))
        ns  = 2 ** hl

        # convert pixel ids to lat/lon using same convention as coords_to_hpyidxs:
        # theta = ((90 - lat) / 180) * pi  =>  lat = 90 - degrees(theta)
        # phi   = ((180 + lon) / 360) * 2pi  =>  lon = degrees(phi) - 180
        thetas, phis = _pix2ang3(ns, occupied_cell_ids, nest=True)
        cell_lats    = 90.0 - _np3.degrees(thetas)
        cell_lons    = _np3.degrees(phis) - 180.0

        t_avg_np = target_avg.detach().cpu().float().numpy()  # (n_occupied, C)
        p_avg_np = pred_avg.detach().cpu().float().numpy()
        diff_np  = t_avg_np - p_avg_np

        n_ch    = t_avg_np.shape[1]
        pt_size = max(1.0, min(20.0, 500000.0 / max(len(cell_lats), 1)))

        print(f"[cell_mse plot] {stream_name}  call={_call_count}  "
              f"n_occupied_cells={len(cell_lats)}  "
              f"n_total_cells={num_cells}  "
              f"hl={hl}  ns={ns}")

        for c in range(n_ch):
            t_vals    = t_avg_np[:, c]
            p_vals    = p_avg_np[:, c]
            diff_vals = diff_np[:, c]

            # percentile-based color scale to reveal spatial detail
            p2,  p98 = _np3.percentile(t_vals,    2), _np3.percentile(t_vals,   98)
            d2,  d98 = _np3.percentile(diff_vals, 2), _np3.percentile(diff_vals, 98)
            dabs     = max(abs(d2), abs(d98), 1e-6)

            fig, axes = _plt3.subplots(1, 3, figsize=(18, 6))

            sc0 = axes[0].scatter(
                cell_lons, cell_lats, c=t_vals,
                s=pt_size, cmap='RdBu_r',
                vmin=p2, vmax=p98,
                linewidths=0, rasterized=True,
            )
            axes[0].set_title(
                f'cell avg target  ch={c}  '
                f'n_cells={len(cell_lats)}'
            )
            axes[0].set_xlabel('longitude (deg)')
            axes[0].set_ylabel('latitude (deg)')
            _plt3.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)

            sc1 = axes[1].scatter(
                cell_lons, cell_lats, c=p_vals,
                s=pt_size, cmap='RdBu_r',
                vmin=p2, vmax=p98,
                linewidths=0, rasterized=True,
            )
            axes[1].set_title(f'cell avg pred  ch={c}')
            axes[1].set_xlabel('longitude (deg)')
            _plt3.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)

            sc2 = axes[2].scatter(
                cell_lons, cell_lats, c=diff_vals,
                s=pt_size, cmap='bwr',
                vmin=-dabs, vmax=dabs,
                linewidths=0, rasterized=True,
            )
            axes[2].set_title(f'cell avg target-pred  ch={c}')
            axes[2].set_xlabel('longitude (deg)')
            _plt3.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04)

            for ax in axes:
                ax.set_aspect('equal')
                ax.grid(True, lw=0.3, alpha=0.3)

            _plt3.suptitle(
                f'healpix_cell_mse  stream={stream_name}  ch={c}  '
                f'call={_call_count}  '
                f'n_cells={len(cell_lats)}/{num_cells}  '
                f'hl={hl}  '
                f'p2={p2:.3f}  p98={p98:.3f}  '
                f'max_diff={dabs:.3f}',
                fontsize=10
            )
            _plt3.tight_layout()
            out_path = os.path.join(
                DEBUG_OUT_DIR,
                f'cell_mse_{stream_name}_ch{c}_call{_call_count:06d}.png'
            )
            _plt3.savefig(out_path, dpi=120, bbox_inches='tight')
            _plt3.close(fig)
            print(f"[cell_mse plot] {stream_name} ch={c} "
                  f"call={_call_count} saved to {out_path}")
    # ----------------------------------------------------------------
    # END DEBUG
    # ----------------------------------------------------------------

    # MSE between cell averages
    diff_sq  = (target_avg - pred_avg) ** 2   # (n_occupied, num_channels)
    loss_chs = diff_sq.mean(0)                # (num_channels,)

    if weights_channels is not None:
        loss = torch.mean(loss_chs * weights_channels.to(dev))
    else:
        loss = torch.mean(loss_chs)

    return loss, loss_chs

def global_fft_mse(
    target: torch.Tensor,
    pred: torch.Tensor,
    target_coords_raw: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
    template_path: str = "",
    freq_weight_power: float = 0.0,
    stream_name: str = "",
):
    """
    Global 2D FFT-based spectral MSE using template-based regridding.

    Follows the same regridding approach as global_haar_wavelet_reshape:
      1. Load template latitude/longitude (2D, shape ny x nx from y and x axes)
      2. For each output grid point find the nearest input point (NearestNDInterpolator)
      3. Gather: values_on_grid = values[Isort], reshape to (ny, nx)
      4. Apply 2D real FFT to both target and pred grids
      5. Compute MSE between the complex spectra (real and imaginary parts separately)
         optionally weighted by frequency magnitude raised to freq_weight_power

    The spectral MSE penalises errors at all spatial frequencies simultaneously.
    freq_weight_power controls the frequency weighting:
        0.0  — equal weight to all frequencies (default)
        1.0  — linearly upweight high frequencies (emphasise fine-scale structure)
       -1.0  — linearly downweight high frequencies (emphasise large-scale structure)
        2.0  — quadratic upweight (strongly emphasise fine-scale structure)

    The loss per channel is normalised by the spatial variance of the target field
    so channels with different magnitudes are comparable.

    Template output grid is cached. Isort is recomputed each call because
    input points change batch to batch due to masking.

    Args:
        target            : (num_points, num_channels)
        pred              : (ens_size, num_points, num_channels)
        target_coords_raw : (num_points, 2) — geographic (lat, lon) degrees
        weights_channels  : (num_channels,) or None
        weights_points    : unused, kept for API consistency
        template_path     : path to NetCDF template with 2D latitude/longitude
        freq_weight_power : exponent for frequency-based weighting of spectral MSE
        stream_name       : used for debug prints and plot filtering

    Returns:
        loss     : scalar
        loss_chs : (num_channels,) per-channel loss
    """
    import math as _math_fft
    import numpy as _np_fft

    num_points, num_channels = target.shape
    dev       = target.device
    pred_mean = pred.mean(0)

    # --- load and cache template output grid ---
    _grid_key  = f'_fft_template_grid_{template_path}'
    _shape_key = f'_fft_template_shape_{template_path}'

    template_grid = getattr(global_fft_mse, _grid_key,  None)
    ny_nx         = getattr(global_fft_mse, _shape_key, None)

    if template_grid is None:
        if not template_path:
            _key = f'_fft_no_template_reported_{stream_name}'
            if not getattr(global_fft_mse, _key, False):
                setattr(global_fft_mse, _key, True)
                print(
                    f"[global_fft_mse] stream={stream_name} "
                    f"template_path is empty. Setting loss to zero."
                )
            return (
                torch.tensor(0.0, device=dev, requires_grad=True),
                torch.zeros(num_channels, device=dev),
            )

        try:
            import xarray as _xr_fft

            template  = _xr_fft.open_dataset(template_path)
            olat      = template.latitude.values.flatten()   # (ny*nx,) C-order
            olon      = template.longitude.values.flatten()
            ny        = len(template.y.values)
            nx        = len(template.x.values)

            template_grid = _np_fft.stack([olat, olon], axis=1)  # (ny*nx, 2)
            setattr(global_fft_mse, _grid_key,  template_grid)
            setattr(global_fft_mse, _shape_key, (ny, nx))
            ny_nx = (ny, nx)

            print(
                f"[global_fft_mse] stream={stream_name} "
                f"template loaded: grid=({ny}x{nx})  "
                f"n_output_points={ny*nx}"
            )

        except Exception as _e:
            _key = f'_fft_template_error_reported_{stream_name}'
            if not getattr(global_fft_mse, _key, False):
                setattr(global_fft_mse, _key, True)
                print(
                    f"[global_fft_mse] stream={stream_name} "
                    f"failed to load template '{template_path}': {_e}. "
                    f"Setting loss to zero."
                )
            return (
                torch.tensor(0.0, device=dev, requires_grad=True),
                torch.zeros(num_channels, device=dev),
            )

    ny, nx = ny_nx

    # --- build Isort fresh each call ---
    import scipy.interpolate as _sci_fft

    ilat    = target_coords_raw[:, 0].cpu().numpy()
    ilon    = target_coords_raw[:, 1].cpu().numpy()
    ipoints = _np_fft.concatenate([ilat[:, None], ilon[:, None]], axis=1)

    interpolator = _sci_fft.NearestNDInterpolator(
        ipoints, _np_fft.arange(len(ilat))
    )
    Isort = interpolator(template_grid).astype(int)   # (ny*nx,)

    # --- gather values onto 2D grid ---
    Isort_t    = torch.from_numpy(Isort).long().to(dev)
    t_flat     = target.float()[Isort_t]      # (ny*nx, C)
    p_flat     = pred_mean.float()[Isort_t]

    t_grid_raw = t_flat.view(ny, nx, num_channels)   # (ny, nx, C)
    p_grid_raw = p_flat.view(ny, nx, num_channels)

    # --- build frequency weight matrix ---
    # freq_ky: (ny, 1),  freq_kx: (1, nx//2+1)
    # rfft2 output has shape (ny, nx//2+1)
    ky = torch.fft.fftfreq(ny, device=dev)                    # (ny,)
    kx = torch.fft.rfftfreq(nx, device=dev)                   # (nx//2+1,)
    freq_mag = torch.sqrt(
        ky[:, None] ** 2 + kx[None, :] ** 2
    )                                                          # (ny, nx//2+1)
    if freq_weight_power != 0.0:
        freq_weight = (freq_mag + 1e-8) ** freq_weight_power  # avoid zero
    else:
        freq_weight = torch.ones_like(freq_mag)

    # --- debug parameters ---
    DEBUG_FFT_PLOT  = True
    DEBUG_PLOT_EVERY_N = 4096
    DEBUG_OUT_DIR   = "/leonardo_scratch/large/userexternal/clussana/wg_debug_fft"
#    DEBUG_ZOOM_LAT_MIN = 57.0
#    DEBUG_ZOOM_LAT_MAX = 62.0
#    DEBUG_ZOOM_LON_MIN = 4.0
#    DEBUG_ZOOM_LON_MAX = 12.0
    # set all four to None for full field:
    DEBUG_ZOOM_LAT_MIN = DEBUG_ZOOM_LAT_MAX = DEBUG_ZOOM_LON_MIN = DEBUG_ZOOM_LON_MAX = None

    _counter_key = '_global_fft_call_count'
    _call_count  = getattr(global_fft_mse, _counter_key, 0)
    if stream_name == "NORA3":
        setattr(global_fft_mse, _counter_key, _call_count + 1)

    lat_min_v = float(template_grid[:, 0].min())
    lat_max_v = float(template_grid[:, 0].max())
    lon_min_v = float(template_grid[:, 1].min())
    lon_max_v = float(template_grid[:, 1].max())

    # --- plot gridded field ---
    if DEBUG_FFT_PLOT \
            and stream_name == "NORA3" \
            and _call_count % DEBUG_PLOT_EVERY_N == 0:

        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as _plt_fft

        os.makedirs(DEBUG_OUT_DIR, exist_ok=True)

        _do_zoom = all(v is not None for v in [
            DEBUG_ZOOM_LAT_MIN, DEBUG_ZOOM_LAT_MAX,
            DEBUG_ZOOM_LON_MIN, DEBUG_ZOOM_LON_MAX,
        ])

        for c in range(num_channels):
            t_np    = t_grid_raw[:, :, c].detach().cpu().float().numpy()
            p_np    = p_grid_raw[:, :, c].detach().cpu().float().numpy()
            diff_np = t_np - p_np

            if _do_zoom:
                _lat_ax = _np_fft.linspace(lat_min_v, lat_max_v, ny)
                _lon_ax = _np_fft.linspace(lon_min_v, lon_max_v, nx)
                _r0 = max(0, int(_np_fft.searchsorted(_lat_ax, DEBUG_ZOOM_LAT_MIN)))
                _r1 = min(ny, int(_np_fft.searchsorted(_lat_ax, DEBUG_ZOOM_LAT_MAX)) + 1)
                _c0 = max(0, int(_np_fft.searchsorted(_lon_ax, DEBUG_ZOOM_LON_MIN)))
                _c1 = min(nx, int(_np_fft.searchsorted(_lon_ax, DEBUG_ZOOM_LON_MAX)) + 1)
                t_plot    = t_np[_r0:_r1, _c0:_c1]
                p_plot    = p_np[_r0:_r1, _c0:_c1]
                diff_plot = diff_np[_r0:_r1, _c0:_c1]
                _extent   = [DEBUG_ZOOM_LON_MIN, DEBUG_ZOOM_LON_MAX,
                             DEBUG_ZOOM_LAT_MIN, DEBUG_ZOOM_LAT_MAX]
                _zoom_str = (f'zoom=[{DEBUG_ZOOM_LAT_MIN},{DEBUG_ZOOM_LAT_MAX}]N '
                             f'[{DEBUG_ZOOM_LON_MIN},{DEBUG_ZOOM_LON_MAX}]E')
            else:
                t_plot    = t_np
                p_plot    = p_np
                diff_plot = diff_np
                _extent   = [lon_min_v, lon_max_v, lat_min_v, lat_max_v]
                _zoom_str = 'full field'

            if t_plot.size == 0:
                continue

            p2,  p98 = _np_fft.percentile(t_plot,    2), _np_fft.percentile(t_plot,   98)
            d2,  d98 = _np_fft.percentile(diff_plot, 2), _np_fft.percentile(diff_plot, 98)
            dabs     = max(abs(d2), abs(d98), 1e-6)

            _imshow_kw = dict(origin='lower', aspect='auto',
                              interpolation='nearest', extent=_extent)

            # --- row 0: spatial field ---
            fig, axes = _plt_fft.subplots(2, 3, figsize=(18, 10))

            im0 = axes[0, 0].imshow(t_plot, cmap='RdBu_r',
                                    vmin=p2, vmax=p98, **_imshow_kw)
            axes[0, 0].set_title(f'target grid  ch={c}  ({ny}x{nx})')
            axes[0, 0].set_xlabel('longitude')
            axes[0, 0].set_ylabel('latitude')
            _plt_fft.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

            im1 = axes[0, 1].imshow(p_plot, cmap='RdBu_r',
                                    vmin=p2, vmax=p98, **_imshow_kw)
            axes[0, 1].set_title(f'pred grid  ch={c}')
            axes[0, 1].set_xlabel('longitude')
            _plt_fft.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

            im2 = axes[0, 2].imshow(diff_plot, cmap='bwr',
                                    vmin=-dabs, vmax=dabs, **_imshow_kw)
            axes[0, 2].set_title(f'target-pred  ch={c}')
            axes[0, 2].set_xlabel('longitude')
            _plt_fft.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

            for ax in axes[0]:
                ax.grid(True, lw=0.3, alpha=0.3)

            # --- row 1: power spectra ---
            t_fft_np = _np_fft.abs(
                _np_fft.fft.rfft2(t_np)
            ) ** 2   # (ny, nx//2+1) power spectrum
            p_fft_np = _np_fft.abs(
                _np_fft.fft.rfft2(p_np)
            ) ** 2

            # 1D power spectrum (average over angles) for comparison
            freq_mag_np = _np_fft.sqrt(
                _np_fft.fft.fftfreq(ny)[:, None] ** 2 +
                _np_fft.fft.rfftfreq(nx)[None, :] ** 2
            ).flatten()
            t_pwr_flat = t_fft_np.flatten()
            p_pwr_flat = p_fft_np.flatten()

            # bin by frequency magnitude
            n_bins     = 50
            freq_edges = _np_fft.linspace(0, freq_mag_np.max(), n_bins + 1)
            t_pwr_bins = _np_fft.zeros(n_bins)
            p_pwr_bins = _np_fft.zeros(n_bins)
            for b in range(n_bins):
                mask = (freq_mag_np >= freq_edges[b]) & (freq_mag_np < freq_edges[b + 1])
                if mask.sum() > 0:
                    t_pwr_bins[b] = t_pwr_flat[mask].mean()
                    p_pwr_bins[b] = p_pwr_flat[mask].mean()
            freq_centres = 0.5 * (freq_edges[:-1] + freq_edges[1:])

            axes[1, 0].semilogy(freq_centres, t_pwr_bins + 1e-10,
                                label='target', color='blue')
            axes[1, 0].semilogy(freq_centres, p_pwr_bins + 1e-10,
                                label='pred',   color='red',  linestyle='--')
            axes[1, 0].set_title(f'1D power spectrum  ch={c}')
            axes[1, 0].set_xlabel('spatial frequency')
            axes[1, 0].set_ylabel('power (log)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, lw=0.3, alpha=0.3)

            # 2D power spectrum target
            im3 = axes[1, 1].imshow(
                _np_fft.log10(t_fft_np + 1e-10),
                origin='lower', aspect='auto', interpolation='nearest',
                cmap='viridis'
            )
            axes[1, 1].set_title(f'log10 power target  ch={c}')
            axes[1, 1].set_xlabel('kx')
            axes[1, 1].set_ylabel('ky')
            _plt_fft.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

            # 2D power spectrum pred
            im4 = axes[1, 2].imshow(
                _np_fft.log10(p_fft_np + 1e-10),
                origin='lower', aspect='auto', interpolation='nearest',
                cmap='viridis'
            )
            axes[1, 2].set_title(f'log10 power pred  ch={c}')
            axes[1, 2].set_xlabel('kx')
            _plt_fft.colorbar(im4, ax=axes[1, 2], fraction=0.046, pad=0.04)

            _plt_fft.suptitle(
                f'global_fft_mse  stream={stream_name}  ch={c}  '
                f'call={_call_count}  grid=({ny}x{nx})  '
                f'{_zoom_str}  '
                f'freq_weight_power={freq_weight_power}',
                fontsize=10
            )
            _plt_fft.tight_layout()
            out_path = os.path.join(
                DEBUG_OUT_DIR,
                f'fft_{stream_name}_ch{c}_call{_call_count:06d}.png'
            )
            _plt_fft.savefig(out_path, dpi=120, bbox_inches='tight')
            _plt_fft.close(fig)
            print(f"[global_fft_mse plot] {stream_name} ch={c} "
                  f"call={_call_count} saved to {out_path}")

    # --- compute FFT loss per channel ---
    loss_chs = torch.zeros(num_channels, device=dev)

    for c in range(num_channels):
        t_field = t_grid_raw[:, :, c]   # (ny, nx)
        p_field = p_grid_raw[:, :, c]

        # 2D real FFT — output shape (ny, nx//2+1) complex
        t_fft = torch.fft.rfft2(t_field)   # (ny, nx//2+1)
        p_fft = torch.fft.rfft2(p_field)

        # MSE on real and imaginary parts separately
        diff_real = (t_fft.real - p_fft.real) ** 2
        diff_imag = (t_fft.imag - p_fft.imag) ** 2
        diff_sq   = diff_real + diff_imag   # (ny, nx//2+1)

        # apply frequency weight
        weighted = diff_sq * freq_weight    # (ny, nx//2+1)

        # mean over all frequency bins
        spectral_mse = weighted.mean()

        # normalise by spatial variance of target channel
        field_var    = t_field.var().clamp(min=1e-8)
        loss_chs[c]  = spectral_mse / field_var

    if weights_channels is not None:
        loss = torch.mean(loss_chs * weights_channels.to(dev))
    else:
        loss = torch.mean(loss_chs)

    return loss, loss_chs


def _reshape_regrid_to_grid(
    target,
    pred_mean,
    target_coords_raw,
    template_grid,
    ny,
    nx,
    num_channels,
    dev,
    regrid_method="nearest",
    stream_name="",
    _report_owner=None,
):
    """
    Regrid a scattered (points, channels) target/pred pair onto the fixed 2D
    template grid, returning flat (ny*nx, C) tensors plus a per-cell validity mask.

    Shared by the *_reshape_varweighted family so the domain-crop fix and the
    nearest/linear regridding option stay identical across them.

    Returns
    -------
    t_flat, p_flat : (ny*nx, C) float tensors gathered onto the template grid.
    valid_cell     : (ny*nx,) bool; False for template cells with no nearby data
                     point (out-of-domain under a regional crop). On those cells
                     p_flat is set equal to t_flat.detach() so every Haar
                     coefficient of (target - pred) vanishes and the cell is
                     loss-neutral.
    empty          : bool; True if NO cell is valid (template/domain mismatch),
                     in which case the caller should return a zero loss.

    regrid_method:
      "nearest" : block replication via NearestNDInterpolator index gather —
                  exact original behaviour.
      "linear"  : barycentric (bilinear-on-a-triangulation) resampling, applied
                  as a differentiable sparse weight matrix to both target and
                  pred. Cells outside the source convex hull fall back to nearest.
    """
    import numpy as _np
    import scipy.interpolate as _sci
    from scipy.spatial import cKDTree as _cKDTree

    owner = _report_owner  # object to hang one-shot warning flags on (a function)

    ilat = target_coords_raw[:, 0].cpu().numpy()
    ilon = target_coords_raw[:, 1].cpu().numpy()
    ipoints = _np.concatenate([ilat[:, None], ilon[:, None]], axis=1)

    # nearest index gather (always computed: it is the default and the fallback)
    interpolator = _sci.NearestNDInterpolator(ipoints, _np.arange(len(ilat)))
    Isort = interpolator(template_grid).astype(int)
    Isort_t = torch.from_numpy(Isort).long().to(dev)

    # --- domain crop: mask template cells with no nearby data point ---
    # template_grid is the FULL (uncropped) grid. Under a regional `domain:` the
    # data covers only a sub-box, and NearestNDInterpolator extrapolates every
    # outside cell to the nearest edge point, smearing it across the sphere.
    # Reject cells whose nearest actual data point is far away. The radius is
    # scaled by the coarser of {template spacing, data spacing} so a legitimately
    # coarse stream (ERA5) on a fine template (MEPS) fills its natural footprint
    # while genuine out-of-domain cells are still rejected.
    _tree_tmpl = _cKDTree(template_grid)
    _tmpl_nn, _ = _tree_tmpl.query(template_grid, k=2)
    _tmpl_dx = float(_np.median(_tmpl_nn[:, 1]))
    if ipoints.shape[0] >= 2:
        _tree_self = _cKDTree(ipoints)
        _self_nn, _ = _tree_self.query(ipoints, k=2)
        _data_dx = float(_np.median(_self_nn[:, 1]))
    else:
        _data_dx = _tmpl_dx
    _max_dist = max(max(_tmpl_dx, _data_dx) * 2.0, 1e-6)

    _tree_data = _cKDTree(ipoints)
    _data_nn, _ = _tree_data.query(template_grid, k=1)
    valid_cell = torch.from_numpy(_data_nn <= _max_dist).to(dev)  # (ny*nx,)

    # nearest gather
    t_flat = target.float()[Isort_t]
    p_flat = pred_mean.float()[Isort_t]

    _valid_c = valid_cell.unsqueeze(-1)
    p_flat = torch.where(_valid_c, p_flat, t_flat.detach())

    # --- optional linear (barycentric) regridding ---
    if regrid_method == "linear" and ipoints.shape[0] >= 3:
        try:
            from scipy.spatial import Delaunay as _Delaunay

            _tri = _Delaunay(ipoints)
            _simplex = _tri.find_simplex(template_grid)
            _inside = _simplex >= 0
            if _inside.any():
                _sx = _simplex[_inside]
                _T = _tri.transform[_sx, :2]
                _r = template_grid[_inside] - _tri.transform[_sx, 2]
                _bary2 = _np.einsum("mij,mj->mi", _T, _r)
                _bary = _np.concatenate(
                    [_bary2, 1.0 - _bary2.sum(axis=1, keepdims=True)], axis=1
                )
                _verts = _tri.simplices[_sx]
                _rows = _np.repeat(_np.flatnonzero(_inside), 3)
                _cols = _verts.reshape(-1)
                _wts = _bary.reshape(-1).astype(_np.float32)
                _W = torch.sparse_coo_tensor(
                    torch.from_numpy(_np.stack([_rows, _cols])).long(),
                    torch.from_numpy(_wts),
                    size=(template_grid.shape[0], ipoints.shape[0]),
                    device=dev,
                    check_invariants=False,
                ).coalesce()
                _t_lin = torch.sparse.mm(_W, target.float())
                _p_lin = torch.sparse.mm(_W, pred_mean.float())
                _inside_t = torch.from_numpy(_inside).to(dev).unsqueeze(-1)
                t_flat = torch.where(_inside_t, _t_lin, t_flat)
                p_flat = torch.where(_inside_t, _p_lin, p_flat)
                p_flat = torch.where(_valid_c, p_flat, t_flat.detach())
        except Exception as _e_lin:
            if owner is not None:
                _key = f"_regrid_linear_failed_{stream_name}"
                if not getattr(owner, _key, False):
                    setattr(owner, _key, True)
                    print(
                        f"[reshape_regrid] stream={stream_name} linear regrid "
                        f"failed ({_e_lin}); falling back to nearest."
                    )
    elif regrid_method not in ("nearest", "linear"):
        if owner is not None:
            _key = f"_regrid_bad_method_{stream_name}"
            if not getattr(owner, _key, False):
                setattr(owner, _key, True)
                print(
                    f"[reshape_regrid] stream={stream_name} unknown "
                    f"regrid_method={regrid_method!r}; using nearest."
                )

    empty = not bool(valid_cell.any())
    return t_flat, p_flat, valid_cell, empty


def global_haar_wavelet_reshape_varweighted(
    target: torch.Tensor,
    pred: torch.Tensor,
    target_coords_raw: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
    template_path: str = "",
    detail_weight: float = 2.0,
    num_levels: int = 3,
    var_weight_epsilon: float = 1e-3,
    stream_name: str = "",
    regrid_method: str = "nearest",
    level_start: int = 0,
):
    """
    Global 2D Haar wavelet MSE with inverse-variance spatial weighting.

    Identical to global_haar_wavelet_reshape except that at each Haar level
    the detail subband MSE is weighted by the inverse of the local target
    variance at that spatial scale. Regions where the target is smooth
    (low local variance, e.g. sea for 2m temperature) receive higher weight,
    penalising prediction errors there more strongly. Regions where the
    target is rough (high local variance, e.g. complex terrain) receive
    lower weight.

    The local variance at level lvl and position (i,j) is estimated directly
    from the Haar detail coefficients of the target at that level:
        local_var[i,j] = t_LH[i,j]**2 + t_HL[i,j]**2 + t_HH[i,j]**2
    This is exact for orthonormal Haar and requires no extra computation.
    The weight is then:
        weight[i,j] = 1.0 / (local_var[i,j] + var_weight_epsilon)
    Weights are normalised by their sum so the total contribution of each
    level is scale-invariant.

    Args:
        target              : (num_points, num_channels)
        pred                : (ens_size, num_points, num_channels)
        target_coords_raw   : (num_points, 2) — geographic (lat, lon) degrees
        weights_channels    : (num_channels,) or None
        weights_points      : unused
        template_path       : path to NetCDF template with 2D lat/lon
        detail_weight       : relative weight of LH/HL/HH vs each other
        num_levels          : number of Haar decomposition levels
        var_weight_epsilon  : floor for local variance to avoid division by
                              zero and to control the maximum upweighting of
                              smooth regions. Larger = less aggressive
                              weighting. Default 1e-3 (normalised units).
        stream_name         : used for debug prints and plot filtering

    Returns:
        loss     : scalar
        loss_chs : (num_channels,) per-channel loss
    """
    import math as _math_vw
    import numpy as _np_vw

    num_points, num_channels = target.shape
    dev       = target.device
    pred_mean = pred.mean(0)

    # --- load and cache template output grid ---
    _grid_key  = f'_vw_template_grid_{template_path}'
    _shape_key = f'_vw_template_shape_{template_path}'

    template_grid = getattr(global_haar_wavelet_reshape_varweighted, _grid_key,  None)
    ny_nx         = getattr(global_haar_wavelet_reshape_varweighted, _shape_key, None)

    if template_grid is None:
        if not template_path:
            _key = f'_vw_no_template_reported_{stream_name}'
            if not getattr(global_haar_wavelet_reshape_varweighted, _key, False):
                setattr(global_haar_wavelet_reshape_varweighted, _key, True)
                print(
                    f"[global_haar_varweighted] stream={stream_name} "
                    f"template_path is empty. Setting loss to zero."
                )
            return (
                torch.tensor(0.0, device=dev, requires_grad=True),
                torch.zeros(num_channels, device=dev),
            )

        try:
            import xarray as _xr_vw

            template  = _xr_vw.open_dataset(template_path)
            olat      = template.latitude.values.flatten()
            olon      = template.longitude.values.flatten()
            ny        = len(template.y.values)
            nx        = len(template.x.values)

            template_grid = _np_vw.stack([olat, olon], axis=1)
            setattr(global_haar_wavelet_reshape_varweighted, _grid_key,  template_grid)
            setattr(global_haar_wavelet_reshape_varweighted, _shape_key, (ny, nx))
            ny_nx = (ny, nx)

            print(
                f"[global_haar_varweighted] stream={stream_name} "
                f"template loaded: grid=({ny}x{nx})  "
                f"n_output_points={ny*nx}"
            )

        except Exception as _e:
            _key = f'_vw_template_error_reported_{stream_name}'
            if not getattr(global_haar_wavelet_reshape_varweighted, _key, False):
                setattr(global_haar_wavelet_reshape_varweighted, _key, True)
                print(
                    f"[global_haar_varweighted] stream={stream_name} "
                    f"failed to load template '{template_path}': {_e}. "
                    f"Setting loss to zero."
                )
            return (
                torch.tensor(0.0, device=dev, requires_grad=True),
                torch.zeros(num_channels, device=dev),
            )

    ny, nx = ny_nx

    # --- regrid scattered points onto template grid (shared helper) ---
    t_flat, p_flat, valid_cell, _empty = _reshape_regrid_to_grid(
        target, pred_mean, target_coords_raw, template_grid, ny, nx,
        num_channels, dev, regrid_method=regrid_method, stream_name=stream_name,
        _report_owner=global_haar_wavelet_reshape_varweighted,
    )
    if _empty:
        _key = f'_vw_empty_domain_reported_{stream_name}'
        if not getattr(global_haar_wavelet_reshape_varweighted, _key, False):
            setattr(global_haar_wavelet_reshape_varweighted, _key, True)
            print(
                f"[global_haar_varweighted] stream={stream_name} no template cell "
                f"near any data point; check template_path vs domain. Loss=0."
            )
        return (
            torch.tensor(0.0, device=dev, requires_grad=True),
            torch.zeros(num_channels, device=dev),
        )

    t_grid_raw = t_flat.view(ny, nx, num_channels)
    p_grid_raw = p_flat.view(ny, nx, num_channels)

    # --- pad to next multiple of 2^num_levels ---
    factor = 2 ** num_levels
    ny_p   = int(_math_vw.ceil(ny / factor)) * factor
    nx_p   = int(_math_vw.ceil(nx / factor)) * factor

    valid_grid_raw = valid_cell.view(ny, nx)  # (ny, nx) bool
    if ny_p > ny or nx_p > nx:
        t_grid = torch.zeros(ny_p, nx_p, num_channels, device=dev)
        p_grid = torch.zeros(ny_p, nx_p, num_channels, device=dev)
        t_grid[:ny, :nx, :] = t_grid_raw
        p_grid[:ny, :nx, :] = p_grid_raw
        # padded region is invalid too (target==pred==0 there, so it is already
        # loss-neutral, but exclude it from the field_var normaliser)
        valid_grid = torch.zeros(ny_p, nx_p, dtype=torch.bool, device=dev)
        valid_grid[:ny, :nx] = valid_grid_raw
    else:
        t_grid = t_grid_raw
        p_grid = p_grid_raw
        valid_grid = valid_grid_raw

    # --- debug parameters ---
    DEBUG_VW_PLOT      = HAAR_DEBUG_ENABLE
    DEBUG_PLOT_EVERY_N = HAAR_DEBUG_EVERY_N
    DEBUG_OUT_DIR      = HAAR_DEBUG_OUT_DIR
#    DEBUG_ZOOM_LAT_MIN = 57.0
#    DEBUG_ZOOM_LAT_MAX = 62.0
#    DEBUG_ZOOM_LON_MIN = 4.0
#    DEBUG_ZOOM_LON_MAX = 12.0
    # set all four to None for full field:
    DEBUG_ZOOM_LAT_MIN = DEBUG_ZOOM_LAT_MAX = DEBUG_ZOOM_LON_MIN = DEBUG_ZOOM_LON_MAX = None

    _counter_key = f'_global_haar_vw_call_count_{stream_name}'
    _call_count  = getattr(global_haar_wavelet_reshape_varweighted, _counter_key, 0)
    if _haar_debug_on(stream_name):
        setattr(global_haar_wavelet_reshape_varweighted, _counter_key, _call_count + 1)

    lat_min_v = float(template_grid[:, 0].min())
    lat_max_v = float(template_grid[:, 0].max())
    lon_min_v = float(template_grid[:, 1].min())
    lon_max_v = float(template_grid[:, 1].max())

    def _apply_zoom_vw(arr_2d):
        if not all(v is not None for v in [
            DEBUG_ZOOM_LAT_MIN, DEBUG_ZOOM_LAT_MAX,
            DEBUG_ZOOM_LON_MIN, DEBUG_ZOOM_LON_MAX,
        ]):
            return arr_2d, [lon_min_v, lon_max_v, lat_min_v, lat_max_v], 'full field'
        _lat_ax = _np_vw.linspace(lat_min_v, lat_max_v, arr_2d.shape[0])
        _lon_ax = _np_vw.linspace(lon_min_v, lon_max_v, arr_2d.shape[1])
        _r0 = max(0, int(_np_vw.searchsorted(_lat_ax, DEBUG_ZOOM_LAT_MIN)))
        _r1 = min(arr_2d.shape[0],
                  int(_np_vw.searchsorted(_lat_ax, DEBUG_ZOOM_LAT_MAX)) + 1)
        _c0 = max(0, int(_np_vw.searchsorted(_lon_ax, DEBUG_ZOOM_LON_MIN)))
        _c1 = min(arr_2d.shape[1],
                  int(_np_vw.searchsorted(_lon_ax, DEBUG_ZOOM_LON_MAX)) + 1)
        _zstr = (f'zoom=[{DEBUG_ZOOM_LAT_MIN},{DEBUG_ZOOM_LAT_MAX}]N '
                 f'[{DEBUG_ZOOM_LON_MIN},{DEBUG_ZOOM_LON_MAX}]E')
        return arr_2d[_r0:_r1, _c0:_c1], \
               [DEBUG_ZOOM_LON_MIN, DEBUG_ZOOM_LON_MAX,
                DEBUG_ZOOM_LAT_MIN, DEBUG_ZOOM_LAT_MAX], _zstr

    # --- plot full gridded field ---
    if DEBUG_VW_PLOT \
            and _haar_debug_on(stream_name) \
            and _call_count % DEBUG_PLOT_EVERY_N == 0:

        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as _plt_vw

        os.makedirs(DEBUG_OUT_DIR, exist_ok=True)

        for c in range(num_channels):
            t_np    = t_grid[:ny, :nx, c].detach().cpu().float().numpy()
            p_np    = p_grid[:ny, :nx, c].detach().cpu().float().numpy()
            diff_np = t_np - p_np

            t_plot,    _ext, _zstr = _apply_zoom_vw(t_np)
            p_plot,    _,    _     = _apply_zoom_vw(p_np)
            diff_plot, _,    _     = _apply_zoom_vw(diff_np)

            if t_plot.size == 0:
                continue

            p2,  p98 = _np_vw.percentile(t_plot,    2), _np_vw.percentile(t_plot,   98)
            d2,  d98 = _np_vw.percentile(diff_plot, 2), _np_vw.percentile(diff_plot, 98)
            dabs     = max(abs(d2), abs(d98), 1e-6)
            _ikw     = dict(origin='lower', aspect='auto',
                            interpolation='nearest', extent=_ext)

            fig, axes = _plt_vw.subplots(1, 3, figsize=(18, 5))
            im0 = axes[0].imshow(t_plot,    cmap='RdBu_r', vmin=p2,   vmax=p98,  **_ikw)
            axes[0].set_title(f'target  ch={c}  ({ny}x{nx})')
            axes[0].set_xlabel('longitude'); axes[0].set_ylabel('latitude')
            _plt_vw.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(p_plot,    cmap='RdBu_r', vmin=p2,   vmax=p98,  **_ikw)
            axes[1].set_title(f'pred  ch={c}')
            axes[1].set_xlabel('longitude')
            _plt_vw.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            im2 = axes[2].imshow(diff_plot, cmap='bwr',    vmin=-dabs, vmax=dabs, **_ikw)
            axes[2].set_title(f'target-pred  ch={c}')
            axes[2].set_xlabel('longitude')
            _plt_vw.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            for ax in axes:
                ax.grid(True, lw=0.3, alpha=0.3)

            _plt_vw.suptitle(
                f'global_haar_varweighted  stream={stream_name}  ch={c}  '
                f'call={_call_count}  grid=({ny}x{nx})  {_zstr}  '
                f'p2={p2:.3f}  p98={p98:.3f}',
                fontsize=10
            )
            _plt_vw.tight_layout()
            out_path = os.path.join(
                DEBUG_OUT_DIR,
                f'haar_vw_{stream_name}_grid_ch{c}_call{_call_count:06d}.png'
            )
            _plt_vw.savefig(out_path, dpi=120, bbox_inches='tight')
            _plt_vw.close(fig)
            print(f"[haar_varweighted grid] {stream_name} ch={c} "
                  f"call={_call_count} saved to {out_path}")

    # --- multi-level Haar per channel ---
    loss_chs = torch.zeros(num_channels, device=dev)

    for c in range(num_channels):
        t_field    = t_grid[:, :, c]
        p_field    = p_grid[:, :, c]
        level_loss = torch.tensor(0.0, device=dev)

        for lvl in range(num_levels):
            t_LL, t_LH, t_HL, t_HH = haar_2d(t_field)
            p_LL, p_LH, p_HL, p_HH = haar_2d(p_field)

            # local variance of target at this level estimated from its own
            # detail coefficients — exact for orthonormal Haar, free to compute
            # shape: (ny/2^(lvl+1), nx/2^(lvl+1))
            local_var = (
                t_LH ** 2 + t_HL ** 2 + t_HH ** 2
            ).detach()   # detach: weight is a constant, not part of gradient

            # inverse variance weight — smooth regions (low var) get high weight
            inv_var_weight = 1.0 / (local_var + var_weight_epsilon)

            # active mask: exclude cells where all detail bands are identical
#            _active_mask = (
#                (t_LH != p_LH) | (t_HL != p_HL) | (t_HH != p_HH)
#            )
            # to remove active_mask
            _active_mask = torch.ones_like(t_LH, dtype=torch.bool)
            _n_active = _active_mask.sum().clamp(min=1)

            if _n_active > 0:
                w     = inv_var_weight[_active_mask]
                w_sum = w.sum().clamp(min=1e-8)

                lh_loss = (((t_LH - p_LH) ** 2)[_active_mask] * w).sum() / w_sum
                hl_loss = (((t_HL - p_HL) ** 2)[_active_mask] * w).sum() / w_sum
                hh_loss = (((t_HH - p_HH) ** 2)[_active_mask] * w).sum() / w_sum

                # only count levels >= level_start (fine fabricated levels are
                # still traversed via the cascade, just not penalised). Default
                # level_start=0 counts every level, unchanged.
                if lvl >= level_start:
                    level_loss = level_loss + (lh_loss + hl_loss + hh_loss) / 3.0

                # ----------------------------------------------------------------
                # DEBUG: plot subbands and inverse variance weight
                # ----------------------------------------------------------------
                if DEBUG_VW_PLOT \
                        and _haar_debug_on(stream_name) \
                        and _call_count % DEBUG_PLOT_EVERY_N == 0:

                    import os
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as _plt_sb

                    os.makedirs(DEBUG_OUT_DIR, exist_ok=True)

                    _full_nr = t_LL.shape[0] * 2
                    _full_nc = t_LL.shape[1] * 2

                    def _zoom_sb(arr):
                        if not all(v is not None for v in [
                            DEBUG_ZOOM_LAT_MIN, DEBUG_ZOOM_LAT_MAX,
                            DEBUG_ZOOM_LON_MIN, DEBUG_ZOOM_LON_MAX,
                        ]):
                            return arr, None, 'full field', 'col', 'row'
                        _la = _np_vw.linspace(lat_min_v, lat_max_v, _full_nr)
                        _lo = _np_vw.linspace(lon_min_v, lon_max_v, _full_nc)
                        _r0s = max(0,
                               int(_np_vw.searchsorted(_la, DEBUG_ZOOM_LAT_MIN)) // 2)
                        _r1s = min(arr.shape[0],
                               int(_np_vw.searchsorted(_la, DEBUG_ZOOM_LAT_MAX)) // 2 + 1)
                        _c0s = max(0,
                               int(_np_vw.searchsorted(_lo, DEBUG_ZOOM_LON_MIN)) // 2)
                        _c1s = min(arr.shape[1],
                               int(_np_vw.searchsorted(_lo, DEBUG_ZOOM_LON_MAX)) // 2 + 1)
                        _ext = [DEBUG_ZOOM_LON_MIN, DEBUG_ZOOM_LON_MAX,
                                DEBUG_ZOOM_LAT_MIN, DEBUG_ZOOM_LAT_MAX]
                        _zs  = (f'zoom=[{DEBUG_ZOOM_LAT_MIN},{DEBUG_ZOOM_LAT_MAX}]N '
                                f'[{DEBUG_ZOOM_LON_MIN},{DEBUG_ZOOM_LON_MAX}]E')
                        return arr[_r0s:_r1s, _c0s:_c1s], _ext, _zs, \
                               'longitude', 'latitude'

                    subbands = {
                        'LL': (t_LL, p_LL),
                        'LH': (t_LH, p_LH),
                        'HL': (t_HL, p_HL),
                        'HH': (t_HH, p_HH),
                    }

                    # plot inverse variance weight map
                    ivw_np = inv_var_weight.detach().cpu().float().numpy()
                    ivw_plot, _ext_ivw, _zstr_ivw, _xl, _yl = _zoom_sb(ivw_np)
                    _ikw_ivw = dict(origin='lower', aspect='auto',
                                    interpolation='nearest')
                    if _ext_ivw is not None:
                        _ikw_ivw['extent'] = _ext_ivw

                    fig, ax = _plt_sb.subplots(1, 1, figsize=(8, 6))
                    im = ax.imshow(_np_vw.log10(ivw_plot + 1e-10),
                                   cmap='hot_r', **_ikw_ivw)
                    ax.set_title(
                        f'log10(inv_var_weight)  ch={c}  lvl={lvl}  '
                        f'eps={var_weight_epsilon}'
                    )
                    ax.set_xlabel(_xl); ax.set_ylabel(_yl)
                    _plt_sb.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.grid(True, lw=0.3, alpha=0.3)
                    _plt_sb.tight_layout()
                    out_path = os.path.join(
                        DEBUG_OUT_DIR,
                        f'haar_vw_{stream_name}_invvar'
                        f'_ch{c}_lvl{lvl}_call{_call_count:06d}.png'
                    )
                    _plt_sb.savefig(out_path, dpi=120, bbox_inches='tight')
                    _plt_sb.close(fig)
                    print(f"[haar_varweighted invvar] ch={c} lvl={lvl} "
                          f"saved to {out_path}")

                    # plot subbands
                    for band_name, (t_b, p_b) in subbands.items():
                        t_b_np   = t_b.detach().cpu().float().numpy()
                        p_b_np   = p_b.detach().cpu().float().numpy()
                        diff_b   = t_b_np - p_b_np

                        t_b_plot,   _ext_sb, _zstr_sb, _xl, _yl = _zoom_sb(t_b_np)
                        p_b_plot,   _,       _,        _,   _   = _zoom_sb(p_b_np)
                        diff_b_plot,_,       _,        _,   _   = _zoom_sb(diff_b)

                        if t_b_plot.size == 0:
                            continue

                        p2b,  p98b = (_np_vw.percentile(t_b_plot,   2),
                                      _np_vw.percentile(t_b_plot,  98))
                        d2b,  d98b = (_np_vw.percentile(diff_b_plot, 2),
                                      _np_vw.percentile(diff_b_plot, 98))
                        dabsb      = max(abs(d2b), abs(d98b), 1e-6)

                        _ikw_sb = dict(origin='lower', aspect='auto',
                                       interpolation='nearest')
                        if _ext_sb is not None:
                            _ikw_sb['extent'] = _ext_sb

                        fig, axes = _plt_sb.subplots(1, 3, figsize=(18, 5))

                        im0 = axes[0].imshow(t_b_plot,    cmap='RdBu_r',
                                             vmin=p2b,    vmax=p98b,   **_ikw_sb)
                        axes[0].set_title(f'target {band_name}  ch={c}  lvl={lvl}')
                        axes[0].set_xlabel(_xl); axes[0].set_ylabel(_yl)
                        _plt_sb.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

                        im1 = axes[1].imshow(p_b_plot,    cmap='RdBu_r',
                                             vmin=p2b,    vmax=p98b,   **_ikw_sb)
                        axes[1].set_title(f'pred {band_name}  ch={c}  lvl={lvl}')
                        axes[1].set_xlabel(_xl)
                        _plt_sb.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                        im2 = axes[2].imshow(diff_b_plot, cmap='bwr',
                                             vmin=-dabsb, vmax=dabsb,  **_ikw_sb)
                        axes[2].set_title(f'target-pred {band_name}  ch={c}  lvl={lvl}')
                        axes[2].set_xlabel(_xl)
                        _plt_sb.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

                        for ax in axes:
                            ax.grid(True, lw=0.3, alpha=0.3)

                        _plt_sb.suptitle(
                            f'haar_varweighted subbands  stream={stream_name}  '
                            f'band={band_name}  ch={c}  lvl={lvl}  '
                            f'call={_call_count}  shape={t_b_plot.shape}  '
                            f'{_zstr_sb}  p2={p2b:.3f}  p98={p98b:.3f}',
                            fontsize=10
                        )
                        _plt_sb.tight_layout()
                        out_path = os.path.join(
                            DEBUG_OUT_DIR,
                            f'haar_vw_{stream_name}_subbands'
                            f'_ch{c}_lvl{lvl}_{band_name}'
                            f'_call{_call_count:06d}.png'
                        )
                        _plt_sb.savefig(out_path, dpi=120, bbox_inches='tight')
                        _plt_sb.close(fig)
                        print(f"[haar_varweighted subband] {stream_name} "
                              f"band={band_name} ch={c} lvl={lvl} "
                              f"shape={t_b_plot.shape} "
                              f"call={_call_count} saved to {out_path}")
                # ----------------------------------------------------------------
                # END DEBUG subbands
                # ----------------------------------------------------------------

            t_field = t_LL
            p_field = p_LL

        # normalise by the target field variance over IN-DOMAIN cells only.
        # Including the masked out-of-domain cells (which hold smeared edge
        # values) would bias this normaliser; restrict to valid_grid. Falls back
        # to the full-grid variance if for some reason nothing is valid.
        _tvals = t_grid[:, :, c]
        if bool(valid_grid.any()):
            field_var = _tvals[valid_grid].var().clamp(min=1e-8)
        else:
            field_var = _tvals.var().clamp(min=1e-8)
        _n_active_lvl = max(num_levels - level_start, 1)
        loss_chs[c] = level_loss / (_n_active_lvl * field_var)

    if weights_channels is not None:
        loss = torch.mean(loss_chs * weights_channels.to(dev))
    else:
        loss = torch.mean(loss_chs)

    return loss, loss_chs


def global_haar_ll_reshape_varweighted(
    target: torch.Tensor,
    pred: torch.Tensor,
    target_coords_raw: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
    template_path: str = "",
    num_levels: int = 3,
    var_weight_epsilon: float = 1e-3,
    stream_name: str = "",
    regrid_method: str = "nearest",
    var_weight_rel: float = 0.0,
    level_start: int = 0,
):
    """
    Global 2D Haar LL-only MSE with inverse-variance spatial weighting,
    using template-based regridding.

    Identical to global_haar_wavelet_reshape_varweighted in structure, but
    the loss is computed only on the LL (approximation) coefficients at each
    Haar level. This penalises errors in the spatially smoothed (low-frequency)
    component of the field at multiple scales:
      - level 0 LL: field averaged at 2x coarser resolution
      - level 1 LL: field averaged at 4x coarser resolution
      - level 2 LL: field averaged at 8x coarser resolution

    The inverse-variance weighting at each level uses the variance of the LL
    field entering that level, estimated as the local variance of the LL
    coefficients across a small neighbourhood. Since LL is a smooth field,
    the local variance is estimated from the detail coefficients at that level:
        local_var_ll[i,j] = t_LH[i,j]**2 + t_HL[i,j]**2 + t_HH[i,j]**2
    This is the same estimator as in global_haar_wavelet_reshape_varweighted
    and is exact for orthonormal Haar. Where the target LL is smooth (low
    local variance, e.g. sea), errors in the LL prediction are penalised more.

    Args:
        target              : (num_points, num_channels)
        pred                : (ens_size, num_points, num_channels)
        target_coords_raw   : (num_points, 2) — geographic (lat, lon) degrees
        weights_channels    : (num_channels,) or None
        weights_points      : unused
        template_path       : path to NetCDF template with 2D lat/lon
        num_levels          : number of Haar decomposition levels
        var_weight_epsilon  : floor for local variance to avoid division by
                              zero. Larger = more uniform weighting.
                              Default 1e-3 (normalised units).
        stream_name         : used for debug prints and plot filtering

    Returns:
        loss     : scalar
        loss_chs : (num_channels,) per-channel loss
    """
    import math as _math_ll
    import numpy as _np_ll

    num_points, num_channels = target.shape
    dev       = target.device
    pred_mean = pred.mean(0)

    # --- load and cache template output grid ---
    _grid_key  = f'_ll_template_grid_{template_path}'
    _shape_key = f'_ll_template_shape_{template_path}'

    template_grid = getattr(global_haar_ll_reshape_varweighted, _grid_key,  None)
    ny_nx         = getattr(global_haar_ll_reshape_varweighted, _shape_key, None)

    if template_grid is None:
        if not template_path:
            _key = f'_ll_no_template_reported_{stream_name}'
            if not getattr(global_haar_ll_reshape_varweighted, _key, False):
                setattr(global_haar_ll_reshape_varweighted, _key, True)
                print(
                    f"[global_haar_ll_reshape_varweighted] stream={stream_name} "
                    f"template_path is empty. Setting loss to zero."
                )
            return (
                torch.tensor(0.0, device=dev, requires_grad=True),
                torch.zeros(num_channels, device=dev),
            )

        try:
            import xarray as _xr_ll

            template  = _xr_ll.open_dataset(template_path)
            olat      = template.latitude.values.flatten()
            olon      = template.longitude.values.flatten()
            ny        = len(template.y.values)
            nx        = len(template.x.values)

            template_grid = _np_ll.stack([olat, olon], axis=1)
            setattr(global_haar_ll_reshape_varweighted, _grid_key,  template_grid)
            setattr(global_haar_ll_reshape_varweighted, _shape_key, (ny, nx))
            ny_nx = (ny, nx)

            print(
                f"[global_haar_ll_reshape_varweighted] stream={stream_name} "
                f"template loaded: grid=({ny}x{nx})  "
                f"n_output_points={ny*nx}"
            )

        except Exception as _e:
            _key = f'_ll_template_error_reported_{stream_name}'
            if not getattr(global_haar_ll_reshape_varweighted, _key, False):
                setattr(global_haar_ll_reshape_varweighted, _key, True)
                print(
                    f"[global_haar_ll_reshape_varweighted] stream={stream_name} "
                    f"failed to load template '{template_path}': {_e}. "
                    f"Setting loss to zero."
                )
            return (
                torch.tensor(0.0, device=dev, requires_grad=True),
                torch.zeros(num_channels, device=dev),
            )

    ny, nx = ny_nx

    # --- regrid scattered points onto template grid (shared helper) ---
    t_flat, p_flat, valid_cell, _empty = _reshape_regrid_to_grid(
        target, pred_mean, target_coords_raw, template_grid, ny, nx,
        num_channels, dev, regrid_method=regrid_method, stream_name=stream_name,
        _report_owner=global_haar_ll_reshape_varweighted,
    )
    if _empty:
        _key = f'_ll_empty_domain_reported_{stream_name}'
        if not getattr(global_haar_ll_reshape_varweighted, _key, False):
            setattr(global_haar_ll_reshape_varweighted, _key, True)
            print(
                f"[global_haar_ll_reshape_varweighted] stream={stream_name} no "
                f"template cell near any data point; check template_path vs domain. Loss=0."
            )
        return (
            torch.tensor(0.0, device=dev, requires_grad=True),
            torch.zeros(num_channels, device=dev),
        )

    t_grid_raw = t_flat.view(ny, nx, num_channels)
    p_grid_raw = p_flat.view(ny, nx, num_channels)

    # --- pad to next multiple of 2^num_levels ---
    factor = 2 ** num_levels
    ny_p   = int(_math_ll.ceil(ny / factor)) * factor
    nx_p   = int(_math_ll.ceil(nx / factor)) * factor

    if ny_p > ny or nx_p > nx:
        t_grid = torch.zeros(ny_p, nx_p, num_channels, device=dev)
        p_grid = torch.zeros(ny_p, nx_p, num_channels, device=dev)
        t_grid[:ny, :nx, :] = t_grid_raw
        p_grid[:ny, :nx, :] = p_grid_raw
    else:
        t_grid = t_grid_raw
        p_grid = p_grid_raw

    # --- debug parameters ---
    DEBUG_LL_PLOT      = HAAR_DEBUG_ENABLE
    DEBUG_PLOT_EVERY_N = HAAR_DEBUG_EVERY_N
    DEBUG_OUT_DIR      = HAAR_DEBUG_OUT_DIR
    DEBUG_ZOOM_LAT_MIN = DEBUG_ZOOM_LAT_MAX = DEBUG_ZOOM_LON_MIN = DEBUG_ZOOM_LON_MAX = None
    # or zoom:
    # DEBUG_ZOOM_LAT_MIN = 57.0
    # DEBUG_ZOOM_LAT_MAX = 62.0
    # DEBUG_ZOOM_LON_MIN = 4.0
    # DEBUG_ZOOM_LON_MAX = 12.0

    _counter_key = f'_global_haar_ll_call_count_{stream_name}'
    _call_count  = getattr(global_haar_ll_reshape_varweighted, _counter_key, 0)
    if _haar_debug_on(stream_name):
        setattr(global_haar_ll_reshape_varweighted, _counter_key, _call_count + 1)

    lat_min_v = float(template_grid[:, 0].min())
    lat_max_v = float(template_grid[:, 0].max())
    lon_min_v = float(template_grid[:, 1].min())
    lon_max_v = float(template_grid[:, 1].max())

    def _apply_zoom_ll(arr_2d):
        if not all(v is not None for v in [
            DEBUG_ZOOM_LAT_MIN, DEBUG_ZOOM_LAT_MAX,
            DEBUG_ZOOM_LON_MIN, DEBUG_ZOOM_LON_MAX,
        ]):
            return arr_2d, [lon_min_v, lon_max_v, lat_min_v, lat_max_v], 'full field'
        _lat_ax = _np_ll.linspace(lat_min_v, lat_max_v, arr_2d.shape[0])
        _lon_ax = _np_ll.linspace(lon_min_v, lon_max_v, arr_2d.shape[1])
        _r0 = max(0, int(_np_ll.searchsorted(_lat_ax, DEBUG_ZOOM_LAT_MIN)))
        _r1 = min(arr_2d.shape[0],
                  int(_np_ll.searchsorted(_lat_ax, DEBUG_ZOOM_LAT_MAX)) + 1)
        _c0 = max(0, int(_np_ll.searchsorted(_lon_ax, DEBUG_ZOOM_LON_MIN)))
        _c1 = min(arr_2d.shape[1],
                  int(_np_ll.searchsorted(_lon_ax, DEBUG_ZOOM_LON_MAX)) + 1)
        _zstr = (f'zoom=[{DEBUG_ZOOM_LAT_MIN},{DEBUG_ZOOM_LAT_MAX}]N '
                 f'[{DEBUG_ZOOM_LON_MIN},{DEBUG_ZOOM_LON_MAX}]E')
        return arr_2d[_r0:_r1, _c0:_c1], \
               [DEBUG_ZOOM_LON_MIN, DEBUG_ZOOM_LON_MAX,
                DEBUG_ZOOM_LAT_MIN, DEBUG_ZOOM_LAT_MAX], _zstr

    # --- plot full gridded field ---
    if DEBUG_LL_PLOT \
            and _haar_debug_on(stream_name) \
            and _call_count % DEBUG_PLOT_EVERY_N == 0:

        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as _plt_ll

        os.makedirs(DEBUG_OUT_DIR, exist_ok=True)

        for c in range(num_channels):
            t_np    = t_grid[:ny, :nx, c].detach().cpu().float().numpy()
            p_np    = p_grid[:ny, :nx, c].detach().cpu().float().numpy()
            diff_np = t_np - p_np

            t_plot,    _ext, _zstr = _apply_zoom_ll(t_np)
            p_plot,    _,    _     = _apply_zoom_ll(p_np)
            diff_plot, _,    _     = _apply_zoom_ll(diff_np)

            if t_plot.size == 0:
                continue

            p2,  p98 = _np_ll.percentile(t_plot,    2), _np_ll.percentile(t_plot,   98)
            d2,  d98 = _np_ll.percentile(diff_plot, 2), _np_ll.percentile(diff_plot, 98)
            dabs     = max(abs(d2), abs(d98), 1e-6)
            _ikw     = dict(origin='lower', aspect='auto',
                            interpolation='nearest', extent=_ext)

            fig, axes = _plt_ll.subplots(1, 3, figsize=(18, 5))

            im0 = axes[0].imshow(t_plot,    cmap='RdBu_r', vmin=p2,   vmax=p98,  **_ikw)
            axes[0].set_title(f'target  ch={c}  ({ny}x{nx})')
            axes[0].set_xlabel('longitude'); axes[0].set_ylabel('latitude')
            _plt_ll.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(p_plot,    cmap='RdBu_r', vmin=p2,   vmax=p98,  **_ikw)
            axes[1].set_title(f'pred  ch={c}')
            axes[1].set_xlabel('longitude')
            _plt_ll.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            im2 = axes[2].imshow(diff_plot, cmap='bwr',    vmin=-dabs, vmax=dabs, **_ikw)
            axes[2].set_title(f'target-pred  ch={c}')
            axes[2].set_xlabel('longitude')
            _plt_ll.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            for ax in axes:
                ax.grid(True, lw=0.3, alpha=0.3)

            _plt_ll.suptitle(
                f'global_haar_ll_reshape_varweighted  stream={stream_name}  ch={c}  '
                f'call={_call_count}  grid=({ny}x{nx})  {_zstr}  '
                f'p2={p2:.3f}  p98={p98:.3f}',
                fontsize=10
            )
            _plt_ll.tight_layout()
            out_path = os.path.join(
                DEBUG_OUT_DIR,
                f'haar_ll_{stream_name}_grid_ch{c}_call{_call_count:06d}.png'
            )
            _plt_ll.savefig(out_path, dpi=120, bbox_inches='tight')
            _plt_ll.close(fig)
            print(f"[haar_ll grid] {stream_name} ch={c} "
                  f"call={_call_count} saved to {out_path}")

    # --- multi-level Haar LL-only loss per channel ---
    loss_chs = torch.zeros(num_channels, device=dev)

    for c in range(num_channels):
        t_field    = t_grid[:, :, c]
        p_field    = p_grid[:, :, c]
        level_loss = torch.tensor(0.0, device=dev)

        for lvl in range(num_levels):
            t_LL, t_LH, t_HL, t_HH = haar_2d(t_field)
            p_LL, p_LH, p_HL, p_HH = haar_2d(p_field)

            # local variance of target at this level estimated from detail
            # coefficients — exact for orthonormal Haar, free to compute
            # shape: (ny/2^(lvl+1), nx/2^(lvl+1))
            local_var = (
                t_LH ** 2 + t_HL ** 2 + t_HH ** 2
            ).detach()

            # Regularisation floor for the inverse-variance weight.
            #
            # A fixed absolute `var_weight_epsilon` does not adapt to the per-level
            # variance scale (local_var grows with level: coarser averaging carries
            # more detail variance). At the default 1e-3 this lets a handful of
            # near-zero-variance cells dominate an entire level — measured as an
            # effective participation of ~28/576 cells at level 1 — so the loss and
            # its gradient collapse onto a few flat patches.
            #
            # `var_weight_rel` (>0) switches to a RELATIVE floor scaled to the median
            # positive local variance at THIS level, which regularises proportionally
            # at every level and cannot collapse. The absolute epsilon is retained as
            # a hard backstop via max(). var_weight_rel == 0.0 reproduces the original
            # fixed-epsilon behaviour exactly (bit-identical).
            if var_weight_rel > 0.0:
                _pos = local_var[local_var > 0]
                _scale = _pos.median() if _pos.numel() > 0 else local_var.new_tensor(1.0)
                _eps_eff = torch.clamp(var_weight_rel * _scale, min=var_weight_epsilon)
            else:
                _eps_eff = var_weight_epsilon

            # inverse variance weight — smooth regions get high weight
            inv_var_weight = 1.0 / (local_var + _eps_eff)

            # weighted MSE on LL coefficients only
            w     = inv_var_weight
            w_sum = w.sum().clamp(min=1e-8)
            ll_loss = ((t_LL - p_LL) ** 2 * w).sum() / w_sum

            # Only accumulate loss for levels >= level_start. Lower levels are
            # still traversed (the cascade below advances t_field/p_field), but
            # their loss is not counted. This lets a COARSE stream (e.g. ERA5,
            # ~100 km) skip the fine levels that are pure interpolation artefact
            # on the fine template grid and constrain only the scales it actually
            # resolves. level_start=0 (default) counts every level, unchanged.
            if lvl >= level_start:
                level_loss = level_loss + ll_loss

            # ----------------------------------------------------------------
            # DEBUG: plot LL subband and inverse variance weight for NORA3
            # ----------------------------------------------------------------
            if DEBUG_LL_PLOT \
                    and _haar_debug_on(stream_name) \
                    and _call_count % DEBUG_PLOT_EVERY_N == 0:

                import os
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as _plt_sb

                os.makedirs(DEBUG_OUT_DIR, exist_ok=True)

                _full_nr = t_LL.shape[0] * 2
                _full_nc = t_LL.shape[1] * 2

                def _zoom_sb_ll(arr):
                    if not all(v is not None for v in [
                        DEBUG_ZOOM_LAT_MIN, DEBUG_ZOOM_LAT_MAX,
                        DEBUG_ZOOM_LON_MIN, DEBUG_ZOOM_LON_MAX,
                    ]):
                        return arr, None, 'full field', 'col', 'row'
                    _la = _np_ll.linspace(lat_min_v, lat_max_v, _full_nr)
                    _lo = _np_ll.linspace(lon_min_v, lon_max_v, _full_nc)
                    _r0s = max(0,
                           int(_np_ll.searchsorted(_la, DEBUG_ZOOM_LAT_MIN)) // 2)
                    _r1s = min(arr.shape[0],
                           int(_np_ll.searchsorted(_la, DEBUG_ZOOM_LAT_MAX)) // 2 + 1)
                    _c0s = max(0,
                           int(_np_ll.searchsorted(_lo, DEBUG_ZOOM_LON_MIN)) // 2)
                    _c1s = min(arr.shape[1],
                           int(_np_ll.searchsorted(_lo, DEBUG_ZOOM_LON_MAX)) // 2 + 1)
                    _ext = [DEBUG_ZOOM_LON_MIN, DEBUG_ZOOM_LON_MAX,
                            DEBUG_ZOOM_LAT_MIN, DEBUG_ZOOM_LAT_MAX]
                    _zs  = (f'zoom=[{DEBUG_ZOOM_LAT_MIN},{DEBUG_ZOOM_LAT_MAX}]N '
                            f'[{DEBUG_ZOOM_LON_MIN},{DEBUG_ZOOM_LON_MAX}]E')
                    return arr[_r0s:_r1s, _c0s:_c1s], _ext, _zs, \
                           'longitude', 'latitude'

                # --- plot LL: target, pred, diff ---
                t_ll_np  = t_LL.detach().cpu().float().numpy()
                p_ll_np  = p_LL.detach().cpu().float().numpy()
                diff_ll  = t_ll_np - p_ll_np

                t_ll_plot,   _ext_ll, _zstr_ll, _xl, _yl = _zoom_sb_ll(t_ll_np)
                p_ll_plot,   _,       _,        _,   _   = _zoom_sb_ll(p_ll_np)
                diff_ll_plot,_,       _,        _,   _   = _zoom_sb_ll(diff_ll)

                if t_ll_plot.size > 0:
                    p2b,  p98b = (_np_ll.percentile(t_ll_plot,   2),
                                  _np_ll.percentile(t_ll_plot,  98))
                    d2b,  d98b = (_np_ll.percentile(diff_ll_plot, 2),
                                  _np_ll.percentile(diff_ll_plot, 98))
                    dabsb      = max(abs(d2b), abs(d98b), 1e-6)

                    _ikw_ll = dict(origin='lower', aspect='auto',
                                   interpolation='nearest')
                    if _ext_ll is not None:
                        _ikw_ll['extent'] = _ext_ll

                    fig, axes = _plt_sb.subplots(1, 3, figsize=(18, 5))

                    im0 = axes[0].imshow(t_ll_plot,    cmap='RdBu_r',
                                         vmin=p2b,     vmax=p98b,    **_ikw_ll)
                    axes[0].set_title(f'target LL  ch={c}  lvl={lvl}  '
                                      f'shape={t_ll_plot.shape}')
                    axes[0].set_xlabel(_xl); axes[0].set_ylabel(_yl)
                    _plt_sb.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

                    im1 = axes[1].imshow(p_ll_plot,    cmap='RdBu_r',
                                         vmin=p2b,     vmax=p98b,    **_ikw_ll)
                    axes[1].set_title(f'pred LL  ch={c}  lvl={lvl}')
                    axes[1].set_xlabel(_xl)
                    _plt_sb.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                    im2 = axes[2].imshow(diff_ll_plot, cmap='bwr',
                                         vmin=-dabsb,  vmax=dabsb,   **_ikw_ll)
                    axes[2].set_title(f'target-pred LL  ch={c}  lvl={lvl}  '
                                      f'll_loss={ll_loss.item():.6f}')
                    axes[2].set_xlabel(_xl)
                    _plt_sb.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

                    for ax in axes:
                        ax.grid(True, lw=0.3, alpha=0.3)

                    _plt_sb.suptitle(
                        f'global_haar_ll_reshape_varweighted  stream={stream_name}  '
                        f'ch={c}  lvl={lvl}  '
                        f'call={_call_count}  shape={t_ll_plot.shape}  '
                        f'{_zstr_ll}  p2={p2b:.3f}  p98={p98b:.3f}  '
                        f'll_loss={ll_loss.item():.6f}',
                        fontsize=10
                    )
                    _plt_sb.tight_layout()
                    out_path = os.path.join(
                        DEBUG_OUT_DIR,
                        f'haar_ll_{stream_name}_LL'
                        f'_ch{c}_lvl{lvl}_call{_call_count:06d}.png'
                    )
                    _plt_sb.savefig(out_path, dpi=120, bbox_inches='tight')
                    _plt_sb.close(fig)
                    print(f"[haar_ll LL] {stream_name} ch={c} lvl={lvl} "
                          f"shape={t_ll_plot.shape} "
                          f"ll_loss={ll_loss.item():.6f} "
                          f"call={_call_count} saved to {out_path}")

                # --- plot inverse variance weight map ---
                ivw_np               = inv_var_weight.detach().cpu().float().numpy()
                ivw_plot, _ext_ivw, _zstr_ivw, _xl_ivw, _yl_ivw = _zoom_sb_ll(ivw_np)

                if ivw_plot.size > 0:
                    _ikw_ivw = dict(origin='lower', aspect='auto',
                                    interpolation='nearest')
                    if _ext_ivw is not None:
                        _ikw_ivw['extent'] = _ext_ivw

                    fig, ax = _plt_sb.subplots(1, 1, figsize=(8, 6))
                    im = ax.imshow(_np_ll.log10(ivw_plot + 1e-10),
                                   cmap='hot_r', **_ikw_ivw)
                    ax.set_title(
                        f'log10(inv_var_weight)  ch={c}  lvl={lvl}  '
                        f'eps={var_weight_epsilon}'
                    )
                    ax.set_xlabel(_xl_ivw); ax.set_ylabel(_yl_ivw)
                    _plt_sb.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.grid(True, lw=0.3, alpha=0.3)
                    _plt_sb.tight_layout()
                    out_path = os.path.join(
                        DEBUG_OUT_DIR,
                        f'haar_ll_{stream_name}_invvar'
                        f'_ch{c}_lvl{lvl}_call{_call_count:06d}.png'
                    )
                    _plt_sb.savefig(out_path, dpi=120, bbox_inches='tight')
                    _plt_sb.close(fig)
                    print(f"[haar_ll invvar] {stream_name} ch={c} lvl={lvl} "
                          f"saved to {out_path}")
            # ----------------------------------------------------------------
            # END DEBUG
            # ----------------------------------------------------------------

            # recurse into LL for next level
            t_field = t_LL
            p_field = p_LL

        # normalise by number of ACTIVE levels and spatial variance of target
        field_var    = t_grid[:, :, c].var().clamp(min=1e-8)
        _n_active = max(num_levels - level_start, 1)
        loss_chs[c] = level_loss / (_n_active * field_var)

    if weights_channels is not None:
        loss = torch.mean(loss_chs * weights_channels.to(dev))
    else:
        loss = torch.mean(loss_chs)

    return loss, loss_chs

def global_haar_wavelet_reshape_varweighted_crps(
    target: torch.Tensor,
    pred: torch.Tensor,
    target_coords_raw: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
    template_path: str = "",
    detail_weight: float = 2.0,  # accepted for interface parity; unused
    num_levels: int = 3,
    var_weight_epsilon: float = 1e-3,
    fair: bool = True,
    normalization: str = "std",  # "std" | "var" | "none"
    ll_weight: float = 0.0,
    stream_name: str = "",
):
    """
    Global 2D Haar wavelet kernel CRPS with inverse-variance spatial
    weighting.

    Args:
        target              : (num_points, num_channels)
        pred                : (ens_size, num_points, num_channels), ens_size > 1
        target_coords_raw   : (num_points, 2) — geographic (lat, lon) degrees
        weights_channels    : (num_channels,) or None
        weights_points      : unused (parity with the MSE variant)
        template_path       : path to NetCDF template with 2D lat/lon
        detail_weight       : unused (parity with the MSE variant)
        num_levels          : number of Haar decomposition levels
        var_weight_epsilon  : floor for the local target variance
        fair                : fair kernel-CRPS spread normalization
        normalization       : per-channel scaling of the accumulated level
                              loss: "std" (default, dimensionally consistent
                              with CRPS), "var" (structural parity with the
                              MSE variant), or "none"
        stream_name         : used for debug prints

    Returns:
        loss     : scalar
        loss_chs : (num_channels,) per-channel loss
    """
    num_points, num_channels = target.shape
    dev = target.device
    ens_size = pred.shape[0]
    assert ens_size > 1, (
        f"[global_haar_varweighted_crps] stream={stream_name}: "
        f"ens_size must be > 1 for CRPS (got {ens_size}). "
        f"Set pred_head.ens_size > 1 in the stream config."
    )
    assert normalization in ("std", "var", "none")

    _self = global_haar_wavelet_reshape_varweighted_crps

    # --- load and cache template output grid (same pattern as MSE variant) ---
    _grid_key = f"_vwc_template_grid_{template_path}"
    _shape_key = f"_vwc_template_shape_{template_path}"

    template_grid = getattr(_self, _grid_key, None)
    ny_nx = getattr(_self, _shape_key, None)

    if template_grid is None:
        if not template_path:
            _key = f"_vwc_no_template_reported_{stream_name}"
            if not getattr(_self, _key, False):
                setattr(_self, _key, True)
                print(
                    f"[global_haar_varweighted_crps] stream={stream_name} "
                    f"template_path is empty. Setting loss to zero."
                )
            return (
                torch.tensor(0.0, device=dev, requires_grad=True),
                torch.zeros(num_channels, device=dev),
            )
        try:
            import xarray as _xr

            template = _xr.open_dataset(template_path)
            olat = template.latitude.values.flatten()
            olon = template.longitude.values.flatten()
            ny = len(template.y.values)
            nx = len(template.x.values)

            template_grid = np.stack([olat, olon], axis=1)
            setattr(_self, _grid_key, template_grid)
            setattr(_self, _shape_key, (ny, nx))
            ny_nx = (ny, nx)

            print(
                f"[global_haar_varweighted_crps] stream={stream_name} "
                f"template loaded: grid=({ny}x{nx})  n_output_points={ny * nx}"
            )
        except Exception as _e:
            _key = f"_vwc_template_error_reported_{stream_name}"
            if not getattr(_self, _key, False):
                setattr(_self, _key, True)
                print(
                    f"[global_haar_varweighted_crps] stream={stream_name} "
                    f"failed to load template '{template_path}': {_e}. "
                    f"Setting loss to zero."
                )
            return (
                torch.tensor(0.0, device=dev, requires_grad=True),
                torch.zeros(num_channels, device=dev),
            )

    ny, nx = ny_nx

    # --- build Isort fresh each call (same as MSE variant) ---
    import scipy.interpolate as _sci

    ilat = target_coords_raw[:, 0].cpu().numpy()
    ilon = target_coords_raw[:, 1].cpu().numpy()
    ipoints = np.concatenate([ilat[:, None], ilon[:, None]], axis=1)

    interpolator = _sci.NearestNDInterpolator(ipoints, np.arange(len(ilat)))
    Isort = interpolator(template_grid).astype(int)
    Isort_t = torch.from_numpy(Isort).long().to(dev)

    # --- padded grid size ---
    factor = 2**num_levels
    ny_p = int(math.ceil(ny / factor)) * factor
    nx_p = int(math.ceil(nx / factor)) * factor

    target_f = target.float()
    pred_f = pred.float()

    # --- multi-level Haar per channel ---
    loss_chs = torch.zeros(num_channels, device=dev)

    for c in range(num_channels):
        # gather this channel onto the 2D grid (target: (ny,nx); pred: (E,ny,nx))
        t_grid_raw = target_f[Isort_t, c].view(ny, nx)
        p_grid_raw = pred_f[:, :, c][:, Isort_t].view(ens_size, ny, nx)

        if ny_p > ny or nx_p > nx:
            t_grid = torch.zeros(ny_p, nx_p, device=dev)
            p_grid = torch.zeros(ens_size, ny_p, nx_p, device=dev)
            t_grid[:ny, :nx] = t_grid_raw
            p_grid[:, :ny, :nx] = p_grid_raw
        else:
            t_grid = t_grid_raw
            p_grid = p_grid_raw

        t_field = t_grid           # (H, W)
        p_field = p_grid           # (E, H, W)
        level_loss = torch.tensor(0.0, device=dev)

        for _lvl in range(num_levels):
            t_LL, t_LH, t_HL, t_HH = haar_2d(t_field)   # (h, w) each
            p_LL, p_LH, p_HL, p_HH = haar_2d(p_field)   # (E, h, w) each

            # local variance of target at this level from its own detail
            # coefficients — exact for orthonormal Haar (identical to MSE
            # variant; detached: the weight is a constant)
            local_var = (t_LH**2 + t_HL**2 + t_HH**2).detach()
            inv_var_weight = 1.0 / (local_var + var_weight_epsilon)
            w_sum = inv_var_weight.sum().clamp(min=1e-8)

            # per-coefficient kernel CRPS on each detail subband,
            # inverse-variance weighted (this line replaces
            # ((t_X - p_X) ** 2 * w).sum() / w_sum of the MSE variant)
            lh_loss = (crps_kernel_pointwise(t_LH, p_LH, fair) * inv_var_weight).sum() / w_sum
            hl_loss = (crps_kernel_pointwise(t_HL, p_HL, fair) * inv_var_weight).sum() / w_sum
            hh_loss = (crps_kernel_pointwise(t_HH, p_HH, fair) * inv_var_weight).sum() / w_sum

            level_loss = level_loss + (lh_loss + hl_loss + hh_loss) / 3.0

            # recurse on the approximation band (per member for pred)
            t_field = t_LL
            p_field = p_LL

        # CRPS on the coarsest approximation band (unweighted — it is the
        # large-scale content; inverse-variance weighting is not meaningful here)
        if ll_weight > 0.0:
            ll_loss = crps_kernel_pointwise(t_field, p_field, fair).mean()
            level_loss = level_loss + ll_weight * ll_loss

        # per-channel normalization: CRPS is first-order, so scale by the
        # target field's std by default; "var" reproduces the MSE variant's
        # structure exactly.
        if normalization == "std":
            denom = t_grid.var().clamp(min=1e-8).sqrt()
        elif normalization == "var":
            denom = t_grid.var().clamp(min=1e-8)
        else:
            denom = torch.tensor(1.0, device=dev)

        loss_chs[c] = level_loss / (num_levels * denom)

    if weights_channels is not None:
        loss = torch.mean(loss_chs * weights_channels.to(dev))
    else:
        loss = torch.mean(loss_chs)

    return loss, loss_chs
