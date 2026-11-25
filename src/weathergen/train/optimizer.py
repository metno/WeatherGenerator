# (C) Copyright 2026 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from omegaconf import DictConfig


class AdEMAMix(torch.optim.Optimizer):
    """Three-EMA optimizer from https://arxiv.org/abs/2409.03137.

    Alpha warms up linearly; beta3 warms up linearly in EMA half-life,
    matching Anemoi's AdEMAMix scheduling. Warmups count updates per parameter.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, ...] = (0.9, 0.95, 0.9999),
        alpha: float = 8.0,
        beta3_warmup: int | None = None,
        alpha_warmup: int | None = None,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        for name, value in (
            ("lr", lr),
            ("alpha", alpha),
            ("weight_decay", weight_decay),
            ("eps", eps),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        if len(betas) != 3 or any(not 0 <= beta < 1 for beta in betas):
            raise ValueError(f"AdEMAMix requires three betas in [0, 1), got {betas}")
        for name, steps in (("beta3_warmup", beta3_warmup), ("alpha_warmup", alpha_warmup)):
            if steps is not None and (type(steps) is not int or steps <= 0):
                raise ValueError(f"{name} must be a positive integer or None, got {steps}")
        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                alpha=alpha,
                beta3_warmup=beta3_warmup,
                alpha_warmup=alpha_warmup,
                weight_decay=weight_decay,
                eps=eps,
            ),
        )
        # Checkpoint APIs otherwise initialize empty optimizers with a synthetic update.
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.requires_grad:
                    self._initialize_state(parameter)

    def _initialize_state(self, parameter: torch.Tensor) -> None:
        state = self.state[parameter]
        state["step"] = 0
        for key in ("exp_avg_fast", "exp_avg_slow", "exp_avg_sq"):
            state[key] = torch.zeros_like(parameter)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3_final = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse or parameter.is_complex():
                    raise RuntimeError("AdEMAMix requires dense, real-valued parameters/gradients")
                state = self.state[parameter]
                if not state:
                    self._initialize_state(parameter)
                state["step"] += 1
                step = state["step"]
                alpha = group["alpha"]
                if group["alpha_warmup"] is not None:
                    alpha *= min(1.0, step / group["alpha_warmup"])
                beta3 = beta3_final
                if group["beta3_warmup"] is not None and step < group["beta3_warmup"]:
                    fraction = step / group["beta3_warmup"]
                    # Anemoi regularizes log(beta) by 1e-8, including when beta1 is zero.
                    log_half = math.log(0.5)
                    half_life = log_half * (
                        (1 - fraction) / math.log(beta1 + 1e-8)
                        + fraction / math.log(beta3_final + 1e-8)
                    )
                    beta3 = math.exp(log_half / half_life)

                fast, slow, variance = (
                    state["exp_avg_fast"],
                    state["exp_avg_slow"],
                    state["exp_avg_sq"],
                )
                fast.mul_(beta1).add_(gradient, alpha=1 - beta1)
                slow.mul_(beta3).add_(gradient, alpha=1 - beta3)
                variance.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                denominator = variance.sqrt().div_(math.sqrt(1 - beta2**step))
                denominator.add_(group["eps"])
                update = fast / (1 - beta1**step)
                update.add_(slow, alpha=alpha).div_(denominator)
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        return loss


def build_optimizer(
    model: torch.nn.Module, optimizer_cfg: DictConfig, lr: float, kappa: float
) -> torch.optim.Optimizer:
    """Select an optimizer; preserve legacy AdamW batch-size scaling by default."""
    name = optimizer_cfg.get("name", "adamw").lower()
    if name == "ademamix":
        cfg = optimizer_cfg.ademamix
        return AdEMAMix(
            model.parameters(),
            lr=lr,
            betas=tuple(cfg.betas),
            alpha=cfg.alpha,
            beta3_warmup=cfg.get("beta3_warmup"),
            alpha_warmup=cfg.get("alpha_warmup"),
            weight_decay=optimizer_cfg.weight_decay,
            eps=cfg.eps,
        )
    if name != "adamw":
        raise ValueError(f"Unsupported optimizer {name!r}; expected 'adamw' or 'ademamix'")
    # Legacy SDE scaling: https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    cfg = optimizer_cfg.adamw
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=optimizer_cfg.weight_decay,
        betas=(max(0.5, 1 - kappa * (1 - cfg.beta1)), max(0.9, 1 - kappa * (1 - cfg.beta2))),
        eps=cfg.get("eps", 2e-8) / math.sqrt(kappa),
        fused=True,
    )
