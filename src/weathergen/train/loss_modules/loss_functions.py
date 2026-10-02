# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from functools import lru_cache

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


def crps_kernel_pointwise(
    target: torch.Tensor,
    preds: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Kernel CRPS at each point, with ensemble members on the first axis."""
    ens_size = preds.shape[0]
    if ens_size < 2:
        raise ValueError("Kernel CRPS requires at least two ensemble members.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    skill = torch.abs(preds - target.unsqueeze(0)).mean(dim=0)
    spread = torch.zeros_like(skill)
    for i in range(ens_size - 1):
        spread += torch.abs(preds[i].unsqueeze(0) - preds[i + 1 :]).sum(dim=0)

    spread_weight = alpha / (ens_size * (ens_size - 1)) + (1 - alpha) / ens_size**2
    return skill - spread_weight * spread


@lru_cache(maxsize=8)
def _load_fft_crps_template(template_path: str) -> tuple[np.ndarray, tuple[int, int]]:
    if template_path.endswith(".npz"):
        with np.load(template_path) as template:
            lat = template["lat"].flatten()
            lon = template["lon"].flatten()
            ny, nx = int(template["ny"]), int(template["nx"])
    else:
        import xarray as xr

        with xr.open_dataset(template_path) as template:
            lat = template.latitude.values.flatten()
            lon = template.longitude.values.flatten()
            ny, nx = len(template.y), len(template.x)

    if ny < 2 or nx < 2 or lat.size != ny * nx or lon.size != ny * nx:
        raise ValueError(f"Template coordinates do not match a 2D grid of shape {(ny, nx)}.")
    grid = np.stack((lat, lon), axis=1)
    if not np.isfinite(grid).all():
        raise ValueError("Template latitude/longitude coordinates must be finite.")

    lat_grid = lat.reshape(ny, nx)
    lon_grid = lon.reshape(ny, nx)
    lat_axis = lat_grid[:, 0]
    lon_axis = lon_grid[0, :]
    lat_spacing = np.diff(lat_axis)
    lon_spacing = np.diff(lon_axis)
    if (
        not np.allclose(lat_grid, lat_axis[:, None], rtol=0.0, atol=1e-6)
        or not np.allclose(lon_grid, lon_axis[None, :], rtol=0.0, atol=1e-6)
        or not np.allclose(lat_spacing, lat_spacing[0], rtol=1e-5, atol=1e-8)
        or not np.allclose(lon_spacing, lon_spacing[0], rtol=1e-5, atol=1e-8)
        or lat_spacing[0] == 0
        or lon_spacing[0] == 0
    ):
        raise ValueError(
            "FFT CRPS requires a uniformly spaced rectilinear latitude/longitude grid."
        )
    return grid, (ny, nx)


def _template_point_order(template_grid: np.ndarray, source_points: np.ndarray) -> np.ndarray:
    """Return source indices in template order, requiring an exact grid match."""
    if source_points.shape != template_grid.shape:
        raise ValueError(
            "FFT CRPS only supports complete rectangular grids: source points must "
            "match the template point count."
        )
    if np.allclose(source_points, template_grid, rtol=0.0, atol=1e-5):
        return np.arange(source_points.shape[0])

    from scipy.spatial import cKDTree

    distances, cells = cKDTree(template_grid).query(source_points)
    if np.any(distances > 1e-5) or np.unique(cells).size != template_grid.shape[0]:
        raise ValueError(
            "FFT CRPS only supports complete rectangular grids: each source point "
            "must match exactly one template cell."
        )

    order = np.empty(cells.shape[0], dtype=np.int64)
    order[cells] = np.arange(cells.shape[0])
    return order


def global_fft_crps(
    target: torch.Tensor,
    pred: torch.Tensor,
    target_coords_raw: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
    template_path: str = "",
    alpha: float = 1.0,
    cutoff_ratio: float = 1.0,
    stream_name: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute spectral CRPS on a complete, regular rectangular grid.

    Source points may be flattened or reordered, but must match every template
    cell exactly once. Irregular point clouds are rejected; no interpolation is
    performed.

    ``weights_points`` is accepted for loss-function API compatibility. Spatial
    point weights are not applied because a Fourier coefficient depends on the
    whole grid.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not 0.0 <= cutoff_ratio <= 1.0:
        raise ValueError(f"cutoff_ratio must be in [0, 1], got {cutoff_ratio}")
    if target.ndim != 2 or pred.ndim != 3 or pred.shape[1:] != target.shape:
        raise ValueError(
            "Expected target (points, channels) and pred (ensemble, points, channels)."
        )
    if pred.shape[0] < 2:
        raise ValueError("global_fft_crps requires at least two ensemble members.")
    if target_coords_raw.shape != (target.shape[0], 2):
        raise ValueError("target_coords_raw must have shape (points, 2).")
    if not template_path:
        raise ValueError(f"global_fft_crps for stream {stream_name!r} requires template_path.")

    try:
        template_grid, (ny, nx) = _load_fft_crps_template(template_path)
    except (OSError, KeyError, AttributeError, ValueError) as exc:
        raise ValueError(
            f"Could not load FFT CRPS template {template_path!r} for stream {stream_name!r}: {exc}"
        ) from exc

    coords = target_coords_raw.detach().cpu().numpy()
    if not np.isfinite(coords).all():
        raise ValueError("FFT CRPS requires finite source coordinates.")
    grid_order = _template_point_order(template_grid, coords)
    grid_order = torch.as_tensor(grid_order, device=target.device, dtype=torch.long)
    target_grid_data = target[grid_order].float().reshape(ny, nx, target.shape[-1])
    pred_grid_data = pred[:, grid_order, :].float().reshape(
        pred.shape[0], ny, nx, target.shape[-1]
    )
    valid_grid = torch.isfinite(target_grid_data) & torch.isfinite(pred_grid_data).all(dim=0)

    loss_chs = torch.zeros(target.shape[-1], dtype=torch.float32, device=target.device)
    ky = torch.fft.fftfreq(ny, device=target.device)
    kx = torch.fft.fftfreq(nx, device=target.device)
    frequencies = torch.sqrt(ky[:, None].square() + kx[None, :].square())
    frequency_mask = frequencies < 0.5 * cutoff_ratio

    for channel in range(target.shape[-1]):
        channel_valid = valid_grid[..., channel]
        if not channel_valid.any():
            continue

        target_grid = target_grid_data[..., channel].masked_fill(~channel_valid, 0.0)
        pred_grid = pred_grid_data[..., channel].masked_fill(
            ~channel_valid.unsqueeze(0), 0.0
        )

        target_fft = torch.fft.fft2(target_grid)
        pred_fft = torch.fft.fft2(pred_grid)
        spectral_crps = crps_kernel_pointwise(target_fft, pred_fft, alpha)
        target_std = target_grid[channel_valid].std(unbiased=False).clamp(min=1e-8)
        loss_chs[channel] = (spectral_crps * frequency_mask).mean() / target_std

    if weights_channels is not None:
        weights_channels = weights_channels.to(device=target.device, dtype=loss_chs.dtype)
        loss = (loss_chs * weights_channels).mean()
    else:
        loss = loss_chs.mean()
    return loss, loss_chs


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
