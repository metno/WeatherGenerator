# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Terminal (slurm log) monitoring of multi-level JEPA training and of the stage-3 fine-tuning.

Every line starts with ``[JEPA-MON]`` (warnings: ``[JEPA-MON][WARN]``), so that

    grep JEPA-MON slurm-<jobid>.out          # everything
    grep 'JEPA-MON\\]\\[WARN' slurm-<jobid>.out # only what needs a look

gives the monitoring of a run. Four kinds of output:

1. ``setup``  once at the start: losses / levels / teacher / mask plan / streams / latent
   decoder / init_from sources / frozen and trainable parameters / optimizer groups.
2. ``batch0`` once, on the first training batch: latent shapes of student and teacher per
   level, hidden and teacher-valid fractions per sample, strategies in the batch.
3. ``step``   every ``every`` steps: per level the target spread (collapse), the prediction
   spread, cosine(pred, target), L1 of the prediction vs. L1 of predicting the mean target
   (``skill`` < 1 = better than the mean), effective rank of the targets; the teacher
   (EMA beta and teacher-student distance, or frozen-weights check); gradient norm per
   block; lr per optimizer group; strategy counts.
4. ``loss``   at every terminal log (train and validation): all JEPA / context / masking
   entries of the SSL loss, DDP averaged (JEPA_hl5, context_hl8, frac_hidden_hl8, ...).

Config (all optional), ``train_logging.jepa_monitor``:

    jepa_monitor:
      enabled: true
      every: 100          # steps between `step` reports (default: train_logging.terminal)
      max_tokens: 2048    # tokens per level used for the statistics
      erank: true         # effective rank of the targets (one SVD per level and report)
      collapse_std: 1.0e-3

Only rank 0 prints; nothing here communicates across ranks (with FSDP, parameter and
gradient numbers are those of the rank-0 shard). A failure inside the monitor is reported
once and never stops training.
"""

from __future__ import annotations

import functools
import logging
import math
import re

import numpy as np
import torch
import torch.nn.functional as F

from weathergen.utils.distributed import is_root

logger = logging.getLogger(__name__)

PREFIX = "[JEPA-MON]"
_failed: set[str] = set()


def _safe(fn):
    """Monitoring must never break training: log the first failure of each method."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self.enabled:
            return None
        try:
            return fn(self, *args, **kwargs)
        except Exception as e:  # noqa: BLE001
            if fn.__name__ not in _failed:
                _failed.add(fn.__name__)
                logger.warning(f"{PREFIX}[WARN] monitor {fn.__name__} failed ({e!r}); skipped")
            return None

    return wrapper


def _local(t: torch.Tensor) -> torch.Tensor:
    """Local shard of a DTensor (no communication), the tensor itself otherwise."""
    return t.to_local() if hasattr(t, "to_local") else t


def _block(name: str) -> str:
    """Block of a parameter for the per-block report: encoder.<sub> or the top module."""
    parts = name.removeprefix("module.").split(".")
    if parts[0] == "encoder" and len(parts) > 2:
        return "encoder." + parts[1]
    return parts[0]


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float | np.floating):
        if not math.isfinite(x):
            return str(x)
        return f"{x:.3g}" if abs(x) >= 1e-3 or x == 0.0 else f"{x:.2e}"
    return str(x)


def _cfg_dict(x):
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(x):
            return OmegaConf.to_container(x, resolve=True)
    except Exception:  # noqa: BLE001
        pass
    return x


def _jepa_sample_maps(metadata):
    """(student indices with a JEPA loss, matching teacher indices), as in the SSL loss."""
    source2target, output_info, _, _ = metadata
    sidx = [i for i, info in enumerate(output_info) if "JEPA" in info.global_params["loss"]]
    tidx = [int(np.asarray(source2target[i]).reshape(-1)[0]) for i in sidx]
    return sidx, tidx, output_info


def _strategy(info):
    params = getattr(info, "params", None)
    if params is None or not hasattr(params, "get"):
        return None
    return (params.get("mask_plan", None) or {}).get("strategy", None)


@torch.no_grad()
def representation_stats(pred, tgt, mask, max_tokens=2048, erank=True) -> dict | None:
    """
    Statistics of predictions vs. targets on the selected tokens (mask: [B, N] bool).

    tgt_std  : mean over features of the std across tokens of the targets (-> 0: collapse)
    pred_std : same for the predictions; pred_std / tgt_std -> 0: constant predictions
    cos      : mean cosine similarity between prediction and target
    l1       : mean |pred - target| (what the JEPA loss sees, unweighted)
    skill    : l1 / mean |target - mean target|; < 1 = better than predicting the mean
    erank    : effective rank of the centred targets / feature dim (1 = isotropic)
    """
    p = pred[mask].float()
    t = tgt[mask].float()
    n = p.shape[0]
    if n < 2:
        return None
    if n > max_tokens:
        idx = torch.linspace(0, n - 1, max_tokens, device=p.device).long()
        p, t = p[idx], t[idx]
    tc = t - t.mean(0, keepdim=True)
    out = {
        "n": int(n),
        "tgt_std": t.std(0).mean().item(),
        "pred_std": p.std(0).mean().item(),
        "cos": F.cosine_similarity(p, t, dim=-1).mean().item(),
        "l1": (p - t).abs().mean().item(),
    }
    base = tc.abs().mean().item()
    out["skill"] = out["l1"] / base if base > 0 else float("nan")
    if erank:
        s = torch.linalg.svdvals(tc)
        q = s / s.sum().clamp_min(1e-12)
        ent = -(q * torch.log(q.clamp_min(1e-12))).sum()
        out["erank"] = math.exp(ent.item()) / tc.shape[1]
    return out


class JepaMonitor:
    """See module docstring. One instance per trainer."""

    def __init__(self, cf):
        mcfg = cf.train_logging.get("jepa_monitor", None) or {}
        self.cf = cf
        self.enabled = bool(mcfg.get("enabled", True))
        self.every = int(mcfg.get("every", cf.train_logging.get("terminal", 100)) or 100)
        self.max_tokens = int(mcfg.get("max_tokens", 2048))
        self.erank = bool(mcfg.get("erank", True))
        self.collapse_std = float(mcfg.get("collapse_std", 1e-3))
        self._batch0_done = False
        self._n_reports = 0
        self._first_tgt_std: dict[str, float] = {}
        self._strategy_counts: dict[str, int] = {}
        self._strategy_probs: dict[str, float] = {}
        self._frozen_ref: dict[str, tuple[float, float]] = {}
        self._teacher_ref: dict[str, tuple[float, float]] = {}
        self._warned: set[str] = set()

    # ------------------------------------------------------------------ output ----

    def _info(self, msg: str) -> None:
        for line in msg.split("\n"):
            logger.info(f"{PREFIX} {line}")

    def _warn(self, msg: str, key: str | None = None) -> None:
        """Warn (once per key when a key is given)."""
        if key is not None:
            if key in self._warned:
                return
            self._warned.add(key)
        logger.warning(f"{PREFIX}[WARN] {msg}")

    # ------------------------------------------------------------------ setup -----

    @_safe
    def log_setup(self, model, optimizer, target_aux_calculators, training_cfg) -> None:
        """Once, after model, teacher and optimizer exist (all ranks call it)."""
        if not is_root():
            return
        base = getattr(model, "module", model)
        # reference checksums of frozen weights (student + frozen teacher)
        self._frozen_ref = self._checksums(
            (n, p) for n, p in base.named_parameters() if not p.requires_grad
        )
        teachers = self._teachers(target_aux_calculators)
        for name, t in teachers.items():
            if type(t).__name__ == "FrozenTeacher":
                self._teacher_ref.update(
                    self._checksums(
                        (f"{name}:{n}", p) for n, p in t.teacher_model.named_parameters()
                    )
                )
        from weathergen.datasets.mask_plan import uses_mask_plan

        plan = (
            _cfg_dict(training_cfg.get("mask_plan", None)) or {}
            if uses_mask_plan(training_cfg)
            else {}
        )
        strategies = {
            k: float((v or {}).get("prob", 1.0))
            for k, v in (plan.get("strategies", None) or {}).items()
            if (v or {}).get("enabled", True)
        }
        tot = sum(strategies.values()) or 1.0
        self._strategy_probs = {k: v / tot for k, v in strategies.items()}
        self._strategy_counts = dict.fromkeys(strategies, 0)

        cf = self.cf
        from weathergen.model.encoder import deep_ssl_taps
        from weathergen.model.latent_predictor import jepa_multilevel_cfg

        lines = ["===== setup ====="]
        lines.append(
            f"run_id={cf.general.get('run_id', '?')}  "
            f"training_mode={training_cfg.get('training_mode', '?')}  "
            f"istep={cf.general.get('istep', 0)}"
        )
        losses = _cfg_dict(training_cfg.get("losses", None)) or {}
        for lname, lc in losses.items():
            if not lc.get("enabled", True):
                continue
            fcts = lc.get("loss_fcts", {}) or {}
            desc = []
            for fname, fc in fcts.items():
                w = fc.get("weight", None) if isinstance(fc, dict) else None
                head = fc.get("head", None) if isinstance(fc, dict) else None
                desc.append(f"{fname}(w={w}{', head=' + str(head) if head else ''})")
            lines.append(
                f"loss {lname}: type={lc.get('type')} weight={lc.get('weight', 1.0)} "
                f"terms=[{', '.join(desc)}]"
            )
        if not losses:
            lines.append("losses: none configured")
        ml = jepa_multilevel_cfg(training_cfg)
        if ml is not None:
            lv = ", ".join(
                f"hl{k}: w={v['weight']} out_dim={v['out_dim']}" for k, v in ml["levels"].items()
            )
            lines.append(f"JEPA multi-level head: {lv}; target_norm={ml['target_norm']}")
        if teachers:
            ctx = _cfg_dict(training_cfg.get("context_loss", None))
            lines.append(f"context loss: {ctx if ctx else 'off'}")
            taps = deep_ssl_taps(cf)
            lines.append(
                "deep SSL: "
                + (
                    f"local2global={taps[0]} global layers={sorted(taps[1])}"
                    if taps[0] or taps[1]
                    else "off"
                )
            )
        if not teachers:
            lines.append("teacher: none (no SSL loss: supervised stage)")
        for name, t in teachers.items():
            kind = type(t).__name__
            if kind == "FrozenTeacher":
                n_p = sum(p.numel() for p in t.teacher_model.parameters())
                n_tr = sum(p.numel() for p in t.teacher_model.parameters() if p.requires_grad)
                lines.append(
                    f"teacher [{name}]: FrozenTeacher levels={t.teacher_levels or 'all'} "
                    f"params={n_p:,} trainable={n_tr:,} (must be 0)"
                )
                if n_tr:
                    self._warn(f"frozen teacher [{name}] has {n_tr:,} trainable parameters")
            elif kind == "EMATeacher":
                ema = t.ema_model
                lines.append(
                    f"teacher [{name}]: EMATeacher halflife={getattr(ema, 'halflife_steps', '?')}"
                    f" rampup={getattr(ema, 'rampup_ratio', '?')}"
                )
            else:
                lines.append(f"teacher [{name}]: {kind}")
        if plan:
            for sname, s in (plan.get("strategies", None) or {}).items():
                s = s or {}
                if not s.get("enabled", True):
                    continue
                lines.append(
                    f"mask plan [{sname}] p={self._strategy_probs.get(sname, 0):.2f}: "
                    f"levels={s.get('levels', {})}"
                )
        else:
            lines.append("mask plan: not used")
        for sname, scfg in (cf.get("streams", None) or {}).items():
            lines.append(
                f"stream {sname}: latent_level={scfg.get('latent_level', 'auto')} "
                f"ssl_view={scfg.get('ssl_view', 'both')}"
            )
        ld = _cfg_dict(cf.get("latent_decoder", None))
        if ld and ld.get("enabled", False):
            lines.append(f"latent decoder: {ld}")
        fc = _cfg_dict(cf.get("forecast_cascade", None))
        if fc:
            lines.append(f"forecast_cascade: {fc}")
        lines += self._init_from_lines(base)
        lines += self._param_lines(base)
        lines += self._optimizer_lines(base, optimizer)
        lines.append(
            "legend step reports: tgt_std->0 = collapse | pred/tgt std ratio->0 = constant "
            "predictions | skill<1 = better than predicting the mean | erank = effective "
            "rank / dim"
        )
        lines.append("===== end setup =====")
        self._info("\n".join(lines))

    def _init_from_lines(self, base) -> list[str]:
        from weathergen.model import model_interface

        rep = getattr(model_interface, "INIT_FROM_REPORT", None)
        if not rep:
            return ["init_from: not used"]
        lines = []
        loaded = rep["loaded"]  # param name -> run_id
        per_run: dict[str, dict[str, int]] = {}
        for k, run in loaded.items():
            d = per_run.setdefault(run, {})
            d[_block(k)] = d.get(_block(k), 0) + 1
        for run, blocks in per_run.items():
            desc = ", ".join(f"{b}:{n}" for b, n in sorted(blocks.items()))
            lines.append(f"init_from {run}: {sum(blocks.values())} tensors -> {desc}")
        fresh: dict[str, int] = {}
        for k, _ in base.named_parameters():
            k = k.removeprefix("module.")
            if k not in loaded:
                fresh[_block(k)] = fresh.get(_block(k), 0) + 1
        if fresh:
            desc = ", ".join(f"{b}:{n}" for b, n in sorted(fresh.items()))
            lines.append(f"init_from: freshly initialised (no source) -> {desc}")
        return lines

    def _param_lines(self, base) -> list[str]:
        counts: dict[str, list[int]] = {}
        for name, p in base.named_parameters():
            c = counts.setdefault(_block(name), [0, 0])
            c[0 if p.requires_grad else 1] += p.numel()
        lines = ["parameters per block (trainable / frozen):"]
        for b, c in sorted(counts.items()):
            tag = "  FROZEN" if c[0] == 0 else ("  partly frozen" if c[1] else "")
            lines.append(f"  {b}: {c[0]:,} / {c[1]:,}{tag}")
        tr = sum(c[0] for c in counts.values())
        fr = sum(c[1] for c in counts.values())
        lines.append(f"  total: {tr:,} trainable / {fr:,} frozen")
        if tr == 0:
            self._warn("no trainable parameter in the model")
        return lines

    def _optimizer_lines(self, base, optimizer) -> list[str]:
        names = {id(p): n for n, p in base.named_parameters()}
        lines = []
        for i, g in enumerate(optimizer.param_groups):
            ps = g["params"]
            n_tr = sum(p.numel() for p in ps if p.requires_grad)
            blocks = sorted({_block(names.get(id(p), "?")) for p in ps})
            shown = ", ".join(blocks[:8]) + (", ..." if len(blocks) > 8 else "")
            lines.append(
                f"optimizer group {i} ({g.get('name', 'default' if i == 0 else '')}): "
                f"lr_scale={g.get('lr_scale', 1.0)} wd={g.get('weight_decay')} "
                f"trainable={n_tr:,} blocks=[{shown}]"
            )
        return lines

    # ------------------------------------------------------------------ steps -----

    @staticmethod
    def _teachers(target_aux_calculators) -> dict:
        return {
            k: v
            for k, v in (target_aux_calculators or {}).items()
            if type(v).__name__ in ("EMATeacher", "FrozenTeacher", "EncoderTeacher")
        }

    @staticmethod
    def _checksums(named_params) -> dict[str, tuple[float, float]]:
        """(sum, sum of abs) per tensor, one device->host copy for all tensors."""
        names, vals = [], []
        for n, p in named_params:
            x = _local(p.detach()).float()
            names.append(n)
            vals.append(torch.stack([x.sum(), x.abs().sum()]).cpu())
        if not vals:
            return {}
        flat = torch.stack(vals).tolist()
        return {n: (v[0], v[1]) for n, v in zip(names, flat, strict=True)}

    def should_report(self, istep: int) -> bool:
        return self.enabled and is_root() and istep % self.every == 0

    @_safe
    def on_train_step(
        self,
        istep,
        preds,
        targets_and_auxs,
        metadata,
        loss,
        model,
        optimizer,
        target_aux_calculators,
        ema_step: int | None = None,
    ) -> None:
        """After backward and gradient clipping, before the optimizer step (all ranks)."""
        if not is_root():
            return
        if not torch.isfinite(loss.detach()).all():
            self._warn(f"istep={istep}: loss is {loss.item()} (NaN/Inf)")
        teachers = self._teachers(target_aux_calculators)
        if not self._batch0_done:
            self._batch0_done = True
            self._log_batch0(preds, targets_and_auxs, metadata, teachers)
        self._count_strategies(metadata)
        if istep % self.every != 0:
            return
        self._n_reports += 1
        lines = [f"----- step {istep} -----"]
        lines += self._latent_lines(preds, targets_and_auxs, metadata, teachers)
        lines += self._teacher_lines(istep if ema_step is None else ema_step, model, teachers)
        lines += self._grad_lines(model)
        lines += self._frozen_lines(model)
        lrs = ", ".join(f"g{i}={g['lr']:.2e}" for i, g in enumerate(optimizer.param_groups))
        lines.append(f"lr: {lrs}")
        if self._strategy_counts:
            tot = sum(self._strategy_counts.values()) or 1
            desc = ", ".join(
                f"{k}={v} ({v / tot:.2f} vs p={self._strategy_probs.get(k, 0):.2f})"
                for k, v in self._strategy_counts.items()
            )
            lines.append(f"strategies drawn so far (rank 0): {desc}")
        self._info("\n".join(lines))

    def _count_strategies(self, metadata) -> None:
        if not self._strategy_counts:
            return
        _, _, output_info = _jepa_sample_maps(metadata)
        for info in output_info:
            s = _strategy(info)
            if s is not None:
                self._strategy_counts[s] = self._strategy_counts.get(s, 0) + 1

    def _jepa_pairs(self, preds, targets_and_auxs, teachers):
        """(teacher name, student latent dict, teacher latent dict) of the SSL losses."""
        p_latent = preds.latent[0] if isinstance(preds.latent, list) else preds.latent
        if not isinstance(p_latent, dict) or "JEPA" not in p_latent:
            return []
        out = []
        for name in teachers:
            tgt = targets_and_auxs.get(name, None)
            t_latent = getattr(tgt, "latent", None)
            if isinstance(t_latent, dict) and "JEPA" in t_latent:
                out.append((name, p_latent, t_latent))
        return out

    def _log_batch0(self, preds, targets_and_auxs, metadata, teachers) -> None:
        pairs = self._jepa_pairs(preds, targets_and_auxs, teachers)
        lines = ["----- first batch -----"]
        sidx, tidx, output_info = _jepa_sample_maps(metadata)
        strat = [_strategy(output_info[i]) for i in sidx]
        lines.append(f"student samples with JEPA loss: {len(sidx)}; strategies: {strat}")
        if not pairs:
            lines.append("no JEPA prediction/target pair in this batch (supervised stage?)")
        for name, p_lat, t_lat in pairs:
            p_j, t_j = p_lat["JEPA"], t_lat["JEPA"]
            if not isinstance(p_j, dict):
                lines.append(
                    f"[{name}] single-level JEPA: student {tuple(p_j.shape)} "
                    f"teacher {tuple(t_j.shape)}"
                )
                continue
            hidden = p_lat.get("JEPA_hidden", {}) or {}
            valid = t_lat.get("JEPA_valid", {}) or {}
            for lvl in sorted(set(p_j) | set(t_j)):
                ps = tuple(p_j[lvl].shape) if lvl in p_j else None
                ts = tuple(t_j[lvl].shape) if lvl in t_j else None
                line = f"[{name}] hl{lvl}: student {ps} teacher {ts}"
                if ps and ts and ps[1] != ts[1]:
                    self._warn(
                        f"hl{lvl}: student and teacher have different cell counts "
                        f"{ps[1]} vs {ts[1]}"
                    )
                if lvl in hidden and sidx:
                    h = hidden[lvl][sidx].float().mean(-1)
                    line += (
                        f" | hidden per sample min/mean/max "
                        f"{h.min().item():.2f}/{h.mean().item():.2f}/{h.max().item():.2f}"
                    )
                    if h.max().item() == 0.0:
                        self._warn(
                            f"hl{lvl}: no hidden cell in the first batch "
                            "(nothing to predict at this level)",
                            key=f"nohid{lvl}",
                        )
                v = valid.get(lvl, valid.get(-1, None)) if valid else None
                if v is not None and tidx:
                    vf = v[tidx].float().mean(-1)
                    line += f" | teacher-valid mean {vf.mean().item():.2f}"
                    if vf.max().item() == 0.0:
                        self._warn(
                            f"hl{lvl}: teacher has no valid cell (target stream "
                            "outside the domain or not in the teacher batch?)"
                        )
                lines.append(line)
        self._info("\n".join(lines))

    def _latent_lines(self, preds, targets_and_auxs, metadata, teachers) -> list[str]:
        pairs = self._jepa_pairs(preds, targets_and_auxs, teachers)
        if not pairs:
            return []
        sidx, tidx, _ = _jepa_sample_maps(metadata)
        if not sidx:
            return ["latent: no JEPA sample in this batch"]
        lines = []
        for name, p_lat, t_lat in pairs:
            p_j, t_j = p_lat["JEPA"], t_lat["JEPA"]
            if not isinstance(p_j, dict):
                p_j, t_j = {0: p_j}, {0: t_j}
            hidden = p_lat.get("JEPA_hidden", {}) or {}
            valid = t_lat.get("JEPA_valid", {}) or {}
            for lvl in sorted(set(p_j) & set(t_j)):
                s_x, t_x = p_j[lvl][sidx], t_j[lvl][tidx].to(p_j[lvl].device)
                if s_x.dim() != 3 or s_x.shape != t_x.shape:
                    lines.append(
                        f"[{name}] hl{lvl}: shapes {tuple(s_x.shape)} vs "
                        f"{tuple(t_x.shape)}, no statistics"
                    )
                    continue
                m = torch.ones(s_x.shape[:2], dtype=torch.bool, device=s_x.device)
                if lvl in hidden:
                    m = m & hidden[lvl][sidx].to(s_x.device)
                v = valid.get(lvl, valid.get(-1, None)) if valid else None
                if v is not None:
                    m = m & v[tidx].to(s_x.device)
                st = representation_stats(s_x, t_x, m, self.max_tokens, self.erank)
                key = f"{name}:hl{lvl}"
                if st is None:
                    lines.append(
                        f"[{name}] hl{lvl}: no cell both hidden from the student and valid "
                        "for the teacher in this batch (nothing predicted at this level)"
                    )
                    continue
                ratio = st["pred_std"] / st["tgt_std"] if st["tgt_std"] > 0 else float("nan")
                lines.append(
                    f"[{name}] hl{lvl}: tgt_std={_fmt(st['tgt_std'])} "
                    f"pred_std={_fmt(st['pred_std'])} (ratio {_fmt(ratio)}) "
                    f"cos={_fmt(st['cos'])} l1={_fmt(st['l1'])} skill={_fmt(st['skill'])}"
                    + (f" erank={_fmt(st['erank'])}" if "erank" in st else "")
                    + f" tokens={st['n']}"
                )
                first = self._first_tgt_std.setdefault(key, st["tgt_std"])
                if st["tgt_std"] < self.collapse_std or st["tgt_std"] < 0.05 * first:
                    self._warn(
                        f"{key}: target spread {st['tgt_std']:.2e} (first report "
                        f"{first:.2e}) -> teacher representation collapsing",
                        key=f"collapse:{key}",
                    )
                if ratio < 0.1:
                    self._warn(
                        f"{key}: predictions nearly constant (pred/tgt std {ratio:.3f})",
                        key=f"const:{key}",
                    )
                if self._n_reports > 10 and st["skill"] > 1.05:
                    self._warn(
                        f"{key}: predictor still worse than predicting the mean "
                        f"target (skill {st['skill']:.2f}) after {self._n_reports} "
                        "reports",
                        key=f"skill:{key}",
                    )
        return lines

    def _teacher_lines(self, istep, model, teachers) -> list[str]:
        lines = []
        base = getattr(model, "module", model)
        for name, t in teachers.items():
            kind = type(t).__name__
            if kind == "EMATeacher":
                ema = t.ema_model
                try:
                    beta = t.get_current_beta(istep)
                except Exception:  # noqa: BLE001
                    beta = None
                src = dict(base.named_parameters())
                num, den, k = 0.0, 0.0, 0
                for n, p_t in ema.ema_model.named_parameters():
                    p_s = src.get(n)
                    if p_s is None or not n.startswith("encoder"):
                        continue
                    a, b = _local(p_t.detach()).float(), _local(p_s.detach()).float()
                    if a.shape != b.shape:
                        continue
                    num = num + (a - b.to(a.device)).pow(2).sum()
                    den = den + b.pow(2).sum().to(a.device)
                    k += 1
                    if k >= 64:
                        break
                num, den = float(num), float(den)
                rel = math.sqrt(num / den) if den > 0 else float("nan")
                lines.append(
                    f"teacher [{name}] EMA: beta(decay)={_fmt(beta)} "
                    f"|teacher-student|/|student| (encoder)={_fmt(rel)}"
                )
                if k and rel == 0.0 and self._n_reports > 2:
                    self._warn(
                        f"EMA teacher [{name}] identical to the student", key=f"emasame:{name}"
                    )
            elif kind == "FrozenTeacher":
                now = self._checksums(
                    (f"{name}:{n}", p) for n, p in t.teacher_model.named_parameters()
                )
                changed = [k for k, v in now.items() if self._teacher_ref.get(k, v) != v]
                lines.append(f"teacher [{name}] frozen: weights unchanged = {not changed}")
                if changed:
                    self._warn(
                        f"frozen teacher [{name}] weights changed: {changed[:5]}",
                        key=f"tchg:{name}",
                    )
        return lines

    def _grad_lines(self, model) -> list[str]:
        base = getattr(model, "module", model)
        sq_t: dict[str, list[torch.Tensor]] = {}
        trainable: dict[str, int] = {}
        frozen_with_grad = []
        for n, p in base.named_parameters():
            b = _block(n)
            if not p.requires_grad:
                if p.grad is not None:
                    frozen_with_grad.append(n)
                continue
            trainable[b] = trainable.get(b, 0) + p.numel()
            if p.grad is not None:
                g = _local(p.grad.detach()).float()
                sq_t.setdefault(b, []).append(g.pow(2).sum().reshape(1))
        blocks = sorted(sq_t)
        vals = (
            torch.stack(
                [
                    torch.cat([t.to(sq_t[blocks[0]][0].device) for t in sq_t[b]]).sum()
                    for b in blocks
                ]
            ).tolist()
            if blocks
            else []
        )
        sq = dict(zip(blocks, vals, strict=True))
        items = [f"{b}={_fmt(math.sqrt(v))}" for b, v in sorted(sq.items())]
        lines = ["grad norm per block (after clipping):" + ("" if items else " none")]
        for i in range(0, len(items), 4):
            lines.append("  " + "  ".join(items[i : i + 4]))
        no_grad = sorted(b for b in trainable if sq.get(b, 0.0) == 0.0)
        if no_grad:
            lines.append(f"trainable blocks without gradient this step: {', '.join(no_grad)}")
        if frozen_with_grad:
            self._warn(
                f"frozen parameters received gradients: {frozen_with_grad[:5]}", key="frozen_grad"
            )
        bad = [b for b, v in sq.items() if not math.isfinite(v)]
        if bad:
            self._warn(f"non-finite gradients in {bad}")
        return lines

    def _frozen_lines(self, model) -> list[str]:
        if not self._frozen_ref:
            return []
        base = getattr(model, "module", model)
        now = self._checksums((n, p) for n, p in base.named_parameters() if n in self._frozen_ref)
        changed = [k for k, v in now.items() if self._frozen_ref[k] != v]
        if changed:
            self._warn(f"frozen student weights changed: {changed[:5]}", key="frozen_chg")
        return [
            f"frozen student weights unchanged = {not changed} ({len(self._frozen_ref)} tensors)"
        ]

    # ------------------------------------------------------------------ losses ----

    _SSL_KEY = re.compile(r"(^|_)(JEPA|context|frac_hidden|frac_valid|n_)")

    @_safe
    def log_loss_breakdown(self, stage: str, istep: int, losses_all: dict) -> None:
        """At every terminal log, with the DDP-averaged history (root only)."""
        if not is_root():
            return
        groups: dict[str, list[str]] = {}
        for key, vals in losses_all.items():
            if key.endswith("avg"):
                continue
            parts = key.split(".")
            leaf = parts[-1]
            if not self._SSL_KEY.search(leaf):
                continue
            arr = np.asarray(vals, dtype=np.float64)
            val = float(np.nanmean(arr)) if arr.size and not np.isnan(arr).all() else math.nan
            if not math.isfinite(val):
                self._warn(f"{stage} istep={istep}: {key} = {val}")
            groups.setdefault(".".join(parts[:-1]) or "loss", []).append(f"{leaf}={_fmt(val)}")
        if not groups:
            return
        lines = []
        for g, items in groups.items():
            items = sorted(items, key=_loss_key_order)
            for i in range(0, len(items), 6):
                head = f"{stage} istep={istep} [{g}]" if i == 0 else " " * len(f"{stage} ")
                lines.append(f"{head} " + "  ".join(items[i : i + 6]))
        self._info("\n".join(lines))


def _loss_key_order(item: str) -> tuple:
    """JEPA total, JEPA per level, context, masking fractions, per strategy."""
    k = item.split("=")[0]
    if k == "JEPA":
        rank = 0
    elif k.startswith("JEPA_hl"):
        rank = 1
    elif k.startswith("context"):
        rank = 2
    elif k.startswith("frac_hidden"):
        rank = 3
    elif k.startswith("frac_valid"):
        rank = 4
    elif k.startswith("JEPA_"):
        rank = 5
    elif k.startswith("n_"):
        rank = 6
    else:
        rank = 9
    return (rank, k)
