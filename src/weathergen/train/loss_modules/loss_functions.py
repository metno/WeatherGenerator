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

    diff = torch.where(mask_nan, target, 0) - torch.where(mask_nan, pred, 0)
#    if not torch.isfinite(diff).all() or diff.abs().max() > 1e4:
#            import torch.distributed as _d
#            _r = _d.get_rank() if _d.is_initialized() else 0
#            print(f"NANDBG[{_r}] LOSS diff: max|diff|={diff.abs().max().item():.3e} "
#                  f"finite={torch.isfinite(diff).all().item()} "
#                  f"max|pred|={pred.abs().max().item():.3e} "
#                  f"max|tgt|={target[mask_nan].abs().max().item() if mask_nan.any() else 0:.3e}", flush=True)
    # NaN-safe p-power. torch.pow(|x|, p) has a singular / ill-defined gradient at
    # x == 0 (PowBackward0 returns NaN when an element is an EXACT match, e.g. on
    # the first step when a zero-initialised head makes pred == target). An exact
    # match must contribute ZERO gradient, not NaN. For even integer p we can use
    # plain multiplication, whose gradient is finite everywhere; for odd p we keep
    # the sign via |x|. This avoids torch.pow entirely for the common p_norm in {1,2}.
    if p_norm == 1:
        diff_p = torch.abs(diff)
    elif p_norm == 2:
        diff_p = diff * diff
    else:
        # general integer p: repeated multiplication of |x|, finite gradient at 0.
        a = torch.abs(diff)
        diff_p = a
        for _ in range(p_norm - 1):
            diff_p = diff_p * a
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
            if template_path.endswith(".npz"):
                import numpy as _np_npz
                _z = _np_npz.load(template_path)
                olat = _z["lat"]
                olon = _z["lon"]
                ny = int(_z["ny"])
                nx = int(_z["nx"])
            else:
                import xarray as _xr_vw
                template = _xr_vw.open_dataset(template_path)
                olat = template.latitude.values.flatten()
                olon = template.longitude.values.flatten()
                ny = len(template.y.values)
                nx = len(template.x.values)
                template.close()

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
            if template_path.endswith(".npz"):
                import numpy as _np_npz
                _z = _np_npz.load(template_path)
                olat = _z["lat"]
                olon = _z["lon"]
                ny = int(_z["ny"])
                nx = int(_z["nx"])
            else:
                import xarray as _xr_vw
                template = _xr_vw.open_dataset(template_path)
                olat = template.latitude.values.flatten()
                olon = template.longitude.values.flatten()
                ny = len(template.y.values)
                nx = len(template.x.values)
                template.close()

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
