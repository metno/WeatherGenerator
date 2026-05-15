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
#    Take pairs of adjacent rows. L is the row-average (low frequency — smooth variation between rows). H is the row-difference (high frequency — how much two adjacent rows differ). Both have shape (H//2, W). This is a 1D Haar along the vertical direction.
    L = (field[0::2, :] + field[1::2, :]) * 0.5
    H = (field[0::2, :] - field[1::2, :]) * 0.5
#    Now take pairs of adjacent columns within L and H. This is the 1D Haar along the horizontal direction, applied to both halves. All four outputs have shape (H//2, W//2).
    LL = (L[:, 0::2] + L[:, 1::2]) * 0.5
    LH = (L[:, 0::2] - L[:, 1::2]) * 0.5
    HL = (H[:, 0::2] + H[:, 1::2]) * 0.5
    HH = (H[:, 0::2] - H[:, 1::2]) * 0.5
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

    # --- multi-level Haar per channel ---
    loss_chs = torch.zeros(num_channels, device=dev)
    subband_norm = 1.0 + 3.0 * detail_weight

    for c in range(num_channels):
        t_field = target_grid[:, :, c]
        p_field = pred_grid[:,   :, c]
        level_loss = torch.tensor(0.0, device=dev)

        for _lvl in range(num_levels):
            t_LL, t_LH, t_HL, t_HH = haar_2d(t_field)
            p_LL, p_LH, p_HL, p_HH = haar_2d(p_field)

            level_loss = level_loss + (
                torch.mean((t_LL - p_LL) ** 2)
                + detail_weight * torch.mean((t_LH - p_LH) ** 2)
                + detail_weight * torch.mean((t_HL - p_HL) ** 2)
                + detail_weight * torch.mean((t_HH - p_HH) ** 2)
            ) / subband_norm

            # recurse into LL subband
            t_field = t_LL
            p_field = p_LL

        loss_chs[c] = level_loss / num_levels

    loss = loss_chs.mean()
    return loss, loss_chs.detach()
