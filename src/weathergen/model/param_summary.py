# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Parameter summary of the model, logged once at the start of training and inference.

Every line starts with `[PARAMS]` (grep PARAMS slurm-<jobid>.out). Counts are of the full
model: with FSDP the parameters are sharded DTensors, whose shape is the global one, so the
numbers do not depend on the number of GPUs. Each line gives total / trainable parameters.

Groups (by parameter name):
  encoder    embedders per stream, per latent level: local engine, local-to-global adapter,
             cell queries, query aggregation, global engine; latent cascade
  processor  forecast engines per level, forecast cascade
  adapter    latent adapter (kept stage-2 predictor, stage 3 of the downscaler)
  decoders   per stream: target-coordinate embedding / target token engine / prediction head
  SSL heads  JEPA / DINO / iBOT heads, post-norms, deep-SSL fusion (training only)
  teachers   EMA copy or frozen teacher (separate models, not trained by the optimizer)
"""

from __future__ import annotations

import logging
import math
import re

logger = logging.getLogger(__name__)

PREFIX = "[PARAMS]"

# encoder parts: (finest-level attribute, per-level ModuleDict attribute, label)
_ENCODER_PARTS = [
    ("ae_local_engine", "_ae_local_per_level", "local engine"),
    ("ae_local_global_engine", "_ae_local_global_per_level", "local-to-global adapter"),
    ("q_cells", "_q_cells_param_dict", "cell queries"),
    ("ae_aggregation_engine", "_ae_aggregation_per_level", "query aggregation"),
    ("ae_global_engine", "_ae_global_per_level", "global engine"),
]


def _numel(p) -> int:
    # global shape also for FSDP2 DTensors (p.shape is the unsharded shape)
    return math.prod(p.shape)


def classify(name: str, finest: int | None) -> tuple[str, str]:
    """(group, label) of a parameter name."""
    name = name.removeprefix("module.").removeprefix("_orig_mod.")
    parts = name.split(".")
    top = parts[0]
    fine = f"hl{finest}" if finest is not None else "latent"

    if top == "encoder":
        sub = parts[1] if len(parts) > 1 else ""
        if sub == "embed_engine" and len(parts) > 3 and parts[2] == "embeds":
            return "encoder", f"embedder {parts[3]}"
        if sub == "latent_cascade":
            return "encoder", "latent cascade"
        for fine_attr, lvl_attr, label in _ENCODER_PARTS:
            if sub == fine_attr:
                return "encoder", f"{fine} {label}"
            if sub == lvl_attr and len(parts) > 2:
                return "encoder", f"hl{parts[2]} {label}"
        return "encoder", f"other ({sub})"

    if top == "forecast_engine":
        return "processor", f"{fine} forecast engine"
    if top == "_forecast_per_level" and len(parts) > 1:
        return "processor", f"hl{parts[1]} forecast engine"
    if top == "forecast_cascade":
        return "processor", "forecast cascade"

    if top == "latent_adapter":
        return "adapter", "latent adapter (kept stage-2 predictor)"

    if top in ("embed_target_coords", "target_token_engines", "pred_heads") and len(parts) > 1:
        return "decoders", parts[1]

    if top == "latent_heads" and len(parts) > 1:
        return "ssl", f"{parts[1]} head"
    if top in ("latent_pre_norm", "latent_pre_norm_levels"):
        return "ssl", "post-norms"
    if top == "deep_ssl_fusion":
        return "ssl", "deep-SSL fusion"
    if top == "deep_ssl_level_projections":
        return "ssl", "deep-SSL projections"

    return "other", top


_DECODER_PART = {
    "embed_target_coords": "coords",
    "target_token_engines": "token engine",
    "pred_heads": "head",
}

_GROUPS = [
    ("encoder", "Encoder"),
    ("processor", "Processor (forecast)"),
    ("adapter", "Latent adapter"),
    ("decoders", "Decoders"),
    ("ssl", "SSL heads (training only)"),
    ("other", "Other"),
]


def _fmt(n: int) -> str:
    if n >= 1e9:
        short = f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        short = f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        short = f"{n / 1e3:.1f}k"
    else:
        short = str(n)
    return f"{n:>15,} ({short:>7})"


_PART_ORDER = [label for _, _, label in _ENCODER_PARTS] + ["forecast engine"]


def _sort_key(label: str):
    """Pipeline order: embedders, then per level (coarse to fine) the parts in data-flow
    order, then cascades / others."""
    if label.startswith("embedder"):
        return (0, 0, 0, label)
    m = re.match(r"hl(\d+) (.*)", label)
    if m:
        part = m.group(2)
        idx = _PART_ORDER.index(part) if part in _PART_ORDER else len(_PART_ORDER)
        return (1, int(m.group(1)), idx, label)
    return (2, 0, 0, label)


def parameter_table(model) -> list[str]:
    """Lines of the summary for one model (without prefix)."""
    base = getattr(model, "module", model)
    finest = getattr(base, "healpix_level", None)
    groups: dict[str, dict[str, list[int]]] = {}
    dec_parts: dict[str, dict[str, int]] = {}
    total, trainable = 0, 0
    for name, p in base.named_parameters():
        n = _numel(p)
        g, label = classify(name, finest)
        c = groups.setdefault(g, {}).setdefault(label, [0, 0])
        c[0] += n
        c[1] += n if p.requires_grad else 0
        total += n
        trainable += n if p.requires_grad else 0
        if g == "decoders":
            top = name.removeprefix("module.").split(".")[0]
            d = dec_parts.setdefault(label, {})
            d[_DECODER_PART[top]] = d.get(_DECODER_PART[top], 0) + n

    w = 44
    lines = [
        f"{'':{w}}{'total':>25}   {'trainable':>25}",
        f"{'Model':{w}}{_fmt(total)}   {_fmt(trainable)}",
    ]
    for g, title in _GROUPS:
        if g not in groups:
            continue
        items = groups[g]
        g_tot = sum(v[0] for v in items.values())
        g_tr = sum(v[1] for v in items.values())
        share = 100.0 * g_tot / max(total, 1)
        lines.append(f"{(title + f'  [{share:.1f}%]'):{w}}{_fmt(g_tot)}   {_fmt(g_tr)}")
        for label in sorted(items, key=_sort_key):
            t, tr = items[label]
            text = f"  {label}"
            if g == "decoders":
                parts = dec_parts.get(label, {})
                detail = " / ".join(
                    f"{k} {parts[k]:,}" for k in _DECODER_PART.values() if k in parts
                )
                lines.append(f"{text:{w}}{_fmt(t)}   {_fmt(tr)}")
                lines.append(f"{'':{w}}    ({detail})")
                continue
            lines.append(f"{text[: w - 1]:{w}}{_fmt(t)}   {_fmt(tr)}")
    return lines


def log_parameter_summary(model, cf=None, target_aux_calculators=None) -> None:
    """Log the parameter summary of the model (+ teachers). Call on rank 0."""
    try:
        lines = ["===== model parameters (total / trainable) ====="]
        lines += parameter_table(model)
        for name, calc in (target_aux_calculators or {}).items():
            kind = type(calc).__name__
            teacher = None
            if kind == "FrozenTeacher":
                teacher = getattr(calc, "teacher_model", None)
            elif kind == "EMATeacher":
                ema = getattr(calc, "ema_model", None)
                teacher = getattr(ema, "ema_model", None)
            if teacher is None:
                continue
            n = sum(_numel(p) for p in teacher.parameters())
            lines.append(
                f"Teacher [{name}] {kind}: {n:,} parameters "
                "(separate model, not updated by the optimizer)"
            )
        if cf is not None and cf.get("with_fsdp", False):
            lines.append("FSDP: counts are of the full model (per GPU it holds about 1/world_size)")
        lines.append("=====")
        for line in lines:
            logger.info(f"{PREFIX} {line}")
    except Exception as e:  # noqa: BLE001 (logging only, never stop the run)
        logger.warning(f"{PREFIX} parameter summary failed: {e!r}")
