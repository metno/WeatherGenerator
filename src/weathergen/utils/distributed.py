# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import hashlib
import json
import os
import sys

import torch
import torch.distributed as dist

SYNC_TIMEOUT_SEC = 60 * 60  # 1 hour
_ENSEMBLE_GROUP: dist.ProcessGroup | None = None
_ENSEMBLE_DEBUG_EVENTS: set[str] = set()


def log_ensemble_diagnostics(event: str, **details) -> None:
    if os.environ.get("WEATHERGEN_DEBUG_ENSEMBLE") != "1" or event in _ENSEMBLE_DEBUG_EVENTS:
        return
    _ENSEMBLE_DEBUG_EVENTS.add(event)
    record = {
        "event": event,
        "pid": os.getpid(),
        "rank": get_rank(),
        "world_size": get_world_size(),
        "distributed_initialized": _is_distributed_initialized(),
        "member_group_size": get_ensemble_parallel_size(),
        "member_ranks": (
            dist.get_process_group_ranks(_ENSEMBLE_GROUP)
            if _ENSEMBLE_GROUP is not None
            else None
        ),
        "distributed_source": __file__,
        **details,
    }
    sys.stderr.write("[ensemble-debug] " + json.dumps(record, default=str) + "\n")
    sys.stderr.flush()


class _GatherEnsemble(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predictions, group):
        ctx.group = group
        members = [torch.empty_like(predictions) for _ in range(dist.get_world_size(group))]
        dist.all_gather(members, predictions.contiguous(), group=group)
        return torch.cat(members, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        gradients = grad_output.contiguous().clone()
        dist.all_reduce(gradients, op=dist.ReduceOp.SUM, group=ctx.group)
        local = gradients.chunk(dist.get_world_size(ctx.group), dim=0)[dist.get_rank(ctx.group)]
        return local.contiguous(), None


def _get_mode_config(cf, section, mode_cfg=None) -> dict:
    settings = dict(cf.get("training_config", {}).get(section, {}))
    if mode_cfg is not None:
        settings.update(mode_cfg.get(section, {}))
    elif cf.get("stage") == "inference":
        settings.update(cf.get("validation_config", {}).get(section, {}))
        settings.update(cf.get("test_config", {}).get(section, {}))
    return settings


def get_noise_config(cf, mode_cfg=None) -> dict:
    """Resolve injector settings independently of ensemble execution."""
    return _get_mode_config(cf, "noise", mode_cfg)


def get_ensemble_size(cf, mode_cfg=None) -> int:
    """Resolve the member count; multi-member execution always uses parallel ranks."""
    size = _get_mode_config(cf, "forecast", mode_cfg).get("num_members", 1)
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("forecast.num_members must be a positive integer.")
    return size


def init_ensemble_parallel(cf):
    """Assign contiguous member ranks to each logical data-parallel sample batch."""
    global _ENSEMBLE_GROUP
    _ENSEMBLE_GROUP = None
    size = get_ensemble_size(cf)
    world_size = get_world_size()
    log_ensemble_diagnostics("init.begin", configured_members=size, stage=cf.get("stage"))
    if size > 1:
        if world_size % size:
            raise ValueError("Distributed world size must be divisible by the ensemble size.")
        for start in range(0, world_size, size):
            ranks = list(range(start, start + size))
            group = dist.new_group(ranks=ranks)
            if get_rank() in ranks:
                _ENSEMBLE_GROUP = group
    cf.data_parallel_rank = get_rank() // size
    cf.data_parallel_size = world_size // size
    log_ensemble_diagnostics(
        "init.ready",
        configured_members=size,
        data_parallel_rank=cf.data_parallel_rank,
        data_parallel_size=cf.data_parallel_size,
    )
    return cf


def get_ensemble_parallel_size() -> int:
    return 1 if _ENSEMBLE_GROUP is None else dist.get_world_size(_ENSEMBLE_GROUP)


def is_ensemble_root() -> bool:
    return _ENSEMBLE_GROUP is None or dist.get_rank(_ENSEMBLE_GROUP) == 0


def validate_ensemble_targets(coordinates: torch.Tensor, lengths: list[int]) -> None:
    """Reject different target layouts before any member skips an empty decoder call."""
    if _ENSEMBLE_GROUP is None:
        return
    digest = hashlib.sha256(
        coordinates.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    )
    digest.update(torch.tensor(lengths, dtype=torch.int64).numpy().tobytes())
    signature = torch.tensor(list(digest.digest()), dtype=torch.uint8, device=coordinates.device)
    signatures = [torch.empty_like(signature) for _ in range(get_ensemble_parallel_size())]
    dist.all_gather(signatures, signature, group=_ENSEMBLE_GROUP)
    if any(not torch.equal(signature, other) for other in signatures):
        raise ValueError("Ensemble members must use identical target coordinates and lengths.")


def gather_ensemble_predictions(predictions: torch.Tensor) -> torch.Tensor:
    """Gather members with backward communication; DDP/FSDP averages weight gradients."""
    mode = "grad" if torch.is_grad_enabled() else "no_grad"
    log_ensemble_diagnostics(
        f"gather.{mode}.input",
        prediction_shape=tuple(predictions.shape),
        device=str(predictions.device),
        dtype=str(predictions.dtype),
        requires_grad=predictions.requires_grad,
    )
    if _ENSEMBLE_GROUP is None:
        log_ensemble_diagnostics(f"gather.{mode}.skipped", reason="No member group initialized")
        return predictions
    ndim = torch.tensor([predictions.ndim], dtype=torch.int64, device=predictions.device)
    dimensions = [torch.empty_like(ndim) for _ in range(get_ensemble_parallel_size())]
    dist.all_gather(dimensions, ndim, group=_ENSEMBLE_GROUP)
    if any(not torch.equal(ndim, other) for other in dimensions):
        raise ValueError("Ensemble members must predict identical target shapes.")
    shape = torch.tensor(predictions.shape, dtype=torch.int64, device=predictions.device)
    shapes = [torch.empty_like(shape) for _ in range(get_ensemble_parallel_size())]
    dist.all_gather(shapes, shape, group=_ENSEMBLE_GROUP)
    if any(not torch.equal(shape, other) for other in shapes):
        raise ValueError("Ensemble members must predict identical target shapes.")
    if torch.is_grad_enabled() and predictions.requires_grad:
        gathered = _GatherEnsemble.apply(predictions, _ENSEMBLE_GROUP)
    else:
        members = [torch.empty_like(predictions) for _ in shapes]
        dist.all_gather(members, predictions.contiguous(), group=_ENSEMBLE_GROUP)
        gathered = torch.cat(members, dim=0)
    log_ensemble_diagnostics(
        f"gather.{mode}.output", prediction_shape=tuple(gathered.shape)
    )
    return gathered


def is_root(pg: dist.ProcessGroup | None = None) -> bool:
    """
    Check if the current rank is the root rank (rank 0).

    Args:
        group (ProcessGroup, optional): The process group to work on.
        If None (default), the default process group will be used.
    """
    if not _is_distributed_initialized():
        # If not initialized, it assumed to be in single process mode.
        # TODO: check what should happen if a process group is passed
        return True
    return dist.get_rank(pg) == 0


def _is_distributed_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """
    Get MPI world size

    Returns:
        int: world size
    """
    if not _is_distributed_initialized():
        return 1

    return dist.get_world_size()


def get_rank() -> int:
    """
    Get current rank number

    Returns:
        int: current rank
    """
    if not _is_distributed_initialized():
        return 0

    return dist.get_rank()


def ddp_average(data: torch.Tensor) -> torch.Tensor:
    """
    Average a tensor across DDP ranks

    Params:
        data: tensor to be averaged (arbitrary shape)

    Return :
        tensor with same shape as data, but entries averaged across all DDP ranks
    """
    if _is_distributed_initialized():
        dist.all_reduce(data, op=dist.ReduceOp.AVG)
    return data.cpu()


def all_gather_vlen(tensor: torch.Tensor, group=None) -> list[torch.Tensor]:
    """Gather tensors with the same number of dimensions but different lengths."""

    if not _is_distributed_initialized():
        return [tensor]

    world_size = dist.get_world_size(group=group)

    # Gather lengths first
    shape = torch.as_tensor(tensor.shape, device=tensor.device)
    shapes = [torch.empty_like(shape) for _ in range(world_size)]
    dist.all_gather(shapes, shape, group=group)

    # Gather data
    inputs = [tensor] * world_size
    outputs = [torch.empty(*_shape, dtype=tensor.dtype, device=tensor.device) for _shape in shapes]
    dist.all_to_all(outputs, inputs, group=group)

    return outputs


def all_gather_vdim(tensor: torch.Tensor, group=None) -> list[torch.Tensor]:
    """Gather tensors with different number of dimensions."""

    if not _is_distributed_initialized():
        return [tensor]

    world_size = dist.get_world_size(group=group)

    # Gather shapes first
    shapes = all_gather_vlen(torch.as_tensor(tensor.shape, device=tensor.device), group=group)

    # Gather data
    inputs = [tensor] * world_size
    outputs = [torch.empty(*_shape, dtype=tensor.dtype, device=tensor.device) for _shape in shapes]
    dist.all_to_all(outputs, inputs, group=group)

    return outputs
