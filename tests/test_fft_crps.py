"""Tests for spectral CRPS on complete regular grids."""

import numpy as np
import pytest
import torch

from weathergen.train.loss_modules.loss_functions import (
    crps_kernel_pointwise,
    global_fft_crps,
)


def _write_template(path, n_side=4):
    lat, lon = np.meshgrid(
        np.linspace(40.0, 44.0, n_side),
        np.linspace(5.0, 9.0, n_side),
        indexing="ij",
    )
    np.savez(path, lat=lat, lon=lon, ny=n_side, nx=n_side)
    coords = torch.from_numpy(np.stack((lat.ravel(), lon.ravel()), axis=1))
    return coords


def test_crps_kernel_alpha_interpolates_classical_and_fair():
    target = torch.tensor([0.0])
    preds = torch.tensor([[0.0], [2.0]])

    classical = crps_kernel_pointwise(target, preds, alpha=0.0)
    fair = crps_kernel_pointwise(target, preds, alpha=1.0)

    assert torch.allclose(classical, torch.tensor([0.5]))
    assert torch.allclose(fair, torch.tensor([0.0]))


def test_global_fft_crps_zero_for_identical_ensemble_and_target(tmp_path):
    template_path = tmp_path / "grid.npz"
    coords = _write_template(template_path)
    target = torch.arange(16, dtype=torch.float32).reshape(16, 1)
    pred = target.unsqueeze(0).repeat(3, 1, 1)

    loss, loss_chs = global_fft_crps(
        target,
        pred,
        coords,
        weights_channels=None,
        weights_points=None,
        template_path=str(template_path),
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    assert torch.allclose(loss_chs, torch.zeros_like(loss_chs), atol=1e-7)


def test_global_fft_crps_is_finite_and_differentiable(tmp_path):
    template_path = tmp_path / "grid.npz"
    coords = _write_template(template_path)
    target = torch.sin(torch.arange(16, dtype=torch.float32)).reshape(16, 1)
    pred = (target + 0.1).unsqueeze(0).repeat(2, 1, 1).requires_grad_()

    loss, loss_chs = global_fft_crps(
        target,
        pred,
        coords,
        weights_channels=torch.tensor([2.0]),
        weights_points=None,
        template_path=str(template_path),
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(loss_chs).all()
    assert loss.item() > 0.0
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_global_fft_crps_reorders_shuffled_regular_grid_points(tmp_path):
    template_path = tmp_path / "grid.npz"
    coords = _write_template(template_path)
    target = torch.sin(torch.arange(16, dtype=torch.float32)).reshape(16, 1)
    pred = torch.stack((target + 0.1, target - 0.2))

    loss_ordered, _ = global_fft_crps(
        target,
        pred,
        coords,
        weights_channels=None,
        weights_points=None,
        template_path=str(template_path),
    )
    permutation = torch.randperm(16, generator=torch.Generator().manual_seed(7))
    loss_shuffled, _ = global_fft_crps(
        target[permutation],
        pred[:, permutation],
        coords[permutation],
        weights_channels=None,
        weights_points=None,
        template_path=str(template_path),
    )

    assert torch.allclose(loss_ordered, loss_shuffled, atol=1e-7)


def test_global_fft_crps_rejects_incomplete_grid(tmp_path):
    template_path = tmp_path / "grid.npz"
    coords = _write_template(template_path)
    target = torch.zeros(15, 1)
    pred = torch.zeros(2, 15, 1)

    with pytest.raises(ValueError, match="complete rectangular grids"):
        global_fft_crps(
            target,
            pred,
            coords[:-1],
            weights_channels=None,
            weights_points=None,
            template_path=str(template_path),
        )


def test_global_fft_crps_rejects_irregular_source_points(tmp_path):
    template_path = tmp_path / "grid.npz"
    coords = _write_template(template_path)
    coords[0, 0] += 0.1
    target = torch.zeros(16, 1)
    pred = torch.zeros(2, 16, 1)

    with pytest.raises(ValueError, match="complete rectangular grids"):
        global_fft_crps(
            target,
            pred,
            coords,
            weights_channels=None,
            weights_points=None,
            template_path=str(template_path),
        )


def test_global_fft_crps_rejects_irregular_template(tmp_path):
    template_path = tmp_path / "grid.npz"
    coords = _write_template(template_path)
    with np.load(template_path) as grid:
        lat = grid["lat"].copy()
        lon = grid["lon"].copy()
    lat[2, :] += 0.1
    np.savez(template_path, lat=lat, lon=lon, ny=4, nx=4)
    target = torch.zeros(16, 1)
    pred = torch.zeros(2, 16, 1)

    with pytest.raises(ValueError, match="uniformly spaced rectilinear"):
        global_fft_crps(
            target,
            pred,
            coords,
            weights_channels=None,
            weights_points=None,
            template_path=str(template_path),
        )


def test_global_fft_crps_rejects_single_member():
    target = torch.zeros(4, 1)
    pred = torch.zeros(1, 4, 1)
    coords = torch.zeros(4, 2)

    with pytest.raises(ValueError, match="at least two ensemble members"):
        global_fft_crps(
            target,
            pred,
            coords,
            weights_channels=None,
            weights_points=None,
            template_path="grid.npz",
        )
