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

