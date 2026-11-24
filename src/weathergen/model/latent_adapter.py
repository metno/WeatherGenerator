# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Latent adapter: keep the JEPA predictor of a latent-alignment run (stage 2 of the CARRA2
downscaler) in the supervised fine-tuning (stage 3).

Why. With a FROZEN teacher (the CARRA2 autoencoder encoder) the stage-2 predictor learns
the map from the student's latent (ERA5 + satellites) to the CARRA2 latent coordinates:
the representation the stage-1 CARRA2 decoder can read. Discarding it would discard that
map. With an EMA teacher the predictor is only a pretraining device (I-JEPA / V-JEPA) and
is discarded as usual: the adapter refuses such a run unless `allow_ema_teacher: true`.

Pipeline with the adapter (stage 3):

    encoders -> latent cascade -> forecast levels (+ forecast_cascade)
             -> [adapter: post-norm (+ deep-SSL fusion) -> predictor, nothing masked except
                 cells without student data, exactly as in stage 2]
             -> decoders (the adapted levels replace the decoder input of those levels)

The adapter is a read-out: the forecast state itself is not replaced (multi-step roll-outs
continue from the encoder / forecast latent).

Config (top level of the stage-3 config):

    latent_adapter:
      enabled: true
      run_id: JEPA_RUN       # stage-2 run whose predictor is kept (frozen CARRA2 teacher)
      mini_epoch: -1
      levels: [8]            # levels whose decoder input is the predictor output
                             # (default: every level the stage-2 predictor predicts)
      load: true             # load predictor, post-norms (and deep-SSL fusion) from run_id
      allow_ema_teacher: false

Architecture and weights come from `run_id`: the predictor settings of its JEPA loss are
copied once into `latent_adapter.resolved` (saved with this run's config, so inference and
continued runs do not need the stage-2 run's config any more). The weights are loaded as
an extra `init_from` source (model_interface.load_init_from). Freeze or train the adapter
like any module: its parameters are `latent_adapter.*`.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from weathergen.model.latent_predictor import MultiLevelLatentPredictor
from weathergen.utils.distributed import is_root

logger = logging.getLogger(__name__)

# stage-2 parameter names -> adapter parameter names (`{finest}` = finest latent level)
ADAPTER_RENAME = {
    r"^latent_heads\.JEPA\.": "latent_adapter.head.",
    r"^latent_pre_norm\.": "latent_adapter.post_norm.{finest}.",
    r"^latent_pre_norm_levels\.": "latent_adapter.post_norm.",
    r"^deep_ssl_fusion\.": "latent_adapter.fusion.",
}
ADAPTER_INCLUDE = [
    r"latent_heads\.JEPA\.(?!deep_proj\.).*",  # deep-SSL output projections are not needed
    r"latent_pre_norm\..*",
    r"latent_pre_norm_levels\..*",
    r"deep_ssl_fusion\..*",
]


def latent_adapter_enabled(cf) -> bool:
    la = cf.get("latent_adapter", None)
    return bool(la) and bool(la.get("enabled", False))


def _to_plain(x):
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(x):
            return OmegaConf.to_container(x, resolve=True)
    except Exception:  # noqa: BLE001
        pass
    return x


def _jepa_loss_of_run(run_cf) -> tuple[dict, dict]:
    """(JEPA loss_fcts conf, target_and_aux_calc conf) of the multi-level JEPA loss."""
    for v in (run_cf.training_config.get("losses", None) or {}).values():
        if v.get("type", None) != "LossLatentSSLStudentTeacher" or not v.get("enabled", True):
            continue
        conf = (v.get("loss_fcts", None) or {}).get("JEPA", None)
        if conf is not None and str(conf.get("head", "")).lower() == "multilevel_transformer":
            return _to_plain(conf), _to_plain(v.get("target_and_aux_calc", None) or {})
    raise ValueError(
        "latent_adapter: the run has no multi-level JEPA loss (head multilevel_transformer)"
    )


def resolve_latent_adapter(cf) -> dict:
    """
    Settings of the adapter, read once from the stage-2 run and stored in
    cf.latent_adapter.resolved: {run_id, mini_epoch, loss_conf, out_levels: {lvl: out_dim or
    None}, replace_levels, deep_taps}.
    """
    from weathergen.common.config import load_run_config
    from weathergen.model.encoder import deep_ssl_taps

    la = cf.latent_adapter
    resolved = la.get("resolved", None)
    if resolved:
        return _to_plain(resolved)

    run_id = la["run_id"]
    mini_epoch = la.get("mini_epoch", -1)
    run_cf = load_run_config(run_id, mini_epoch, model_path=None)
    loss_conf, calc = _jepa_loss_of_run(run_cf)
    teachers = set(calc.keys())
    frozen = "FrozenTeacher" in teachers
    if not frozen and not la.get("allow_ema_teacher", False):
        raise ValueError(
            f"latent_adapter: run {run_id} was trained with {sorted(teachers) or 'no'} teacher. "
            "The predictor of an SSL run with an EMA teacher is a pretraining device and is "
            "discarded (disable latent_adapter and load only the encoder with init_from). "
            "Set latent_adapter.allow_ema_teacher: true to keep it anyway."
        )
    levels_cfg = loss_conf.get("levels", None) or {}
    out_levels = {int(k): ((v or {}).get("out_dim", None)) for k, v in levels_cfg.items()}
    replace = [int(x) for x in (la.get("levels", None) or sorted(out_levels))]
    for lvl in replace:
        assert lvl in out_levels, (
            f"latent_adapter.levels: level {lvl} is not predicted by run {run_id} "
            f"(predicted: {sorted(out_levels)})"
        )
    # deep SSL: the predictor input is the fusion of the encoder taps -> same taps needed
    run_taps = deep_ssl_taps(run_cf)
    own_taps = deep_ssl_taps(cf)
    run_taps_n = (run_taps[0], sorted(run_taps[1] or []))
    own_taps_n = (own_taps[0], sorted(own_taps[1] or []))
    deep = bool(run_taps[0] or run_taps[1] is not None)
    if deep and run_taps_n != own_taps_n:
        raise ValueError(
            f"latent_adapter: run {run_id} used deep SSL taps {run_taps_n}; its predictor reads "
            f"their fusion, so this run needs the same training_config.deep_ssl "
            f"(enabled: true, tap_after) -- got {own_taps_n}"
        )
    resolved = {
        "run_id": str(run_id),
        "mini_epoch": int(mini_epoch),
        "teacher": "frozen" if frozen else "ema",
        "loss_conf": loss_conf,
        "out_levels": {str(k): v for k, v in out_levels.items()},
        "replace_levels": replace,
        "deep": deep,
    }
    try:
        from omegaconf import open_dict

        with open_dict(la):
            la.resolved = resolved
    except Exception:  # noqa: BLE001 (plain dict config)
        la["resolved"] = resolved
    if is_root():
        logger.info(
            f"latent_adapter: keeping the predictor of run {run_id} ({resolved['teacher']} "
            f"teacher) for levels {replace}"
        )
    return resolved


class LatentAdapter(nn.Module):
    """Post-norm (+ deep-SSL fusion) + kept multi-level predictor; see module docstring."""

    def __init__(
        self,
        head: MultiLevelLatentPredictor,
        post_norm: nn.ModuleDict,
        fusion: nn.ModuleDict | None,
        replace_levels: list[int],
        num_aux_tokens: int,
    ):
        super().__init__()
        self.head = head
        self.post_norm = post_norm
        self.fusion = fusion
        self.replace_levels = list(replace_levels)
        self.num_aux = num_aux_tokens

    def forward(
        self,
        tokens_lvl: dict[int, torch.Tensor],
        hidden: dict[int, torch.Tensor],
        intermediates: dict | None = None,
    ) -> dict[int, torch.Tensor]:
        """
        tokens_lvl : {level: (B, num_aux + N_l, D_l)} latent after the forecast levels
        hidden     : {level: (B, N_l) bool} cells without student input
        Returns the decoder input: tokens_lvl with the cells of `replace_levels` replaced by
        the predictor output (aux tokens kept, or zeros if the size changes; the decoders
        do not read the aux tokens).
        """
        if self.fusion is not None:
            assert intermediates is not None, (
                "latent_adapter: the kept predictor reads deep-SSL taps (step 0 only)"
            )
            inputs = {
                lvl: self.fusion[str(lvl)](intermediates[lvl] + [t])
                for lvl, t in tokens_lvl.items()
            }
        else:
            inputs = tokens_lvl
        states = {lvl: self.post_norm[str(lvl)](t) for lvl, t in inputs.items()}
        preds, _ = self.head(states, hidden)
        out = dict(tokens_lvl)
        for lvl in self.replace_levels:
            t, p = tokens_lvl[lvl], preds[lvl]
            aux = t[:, : self.num_aux]
            if aux.shape[-1] != p.shape[-1]:
                aux = torch.zeros(*aux.shape[:-1], p.shape[-1], dtype=p.dtype, device=p.device)
            out[lvl] = torch.cat([aux.to(p.dtype), p], dim=1)
        return out

    @torch.no_grad()
    def correction_norms(self, tokens_lvl, adapted) -> dict[int, torch.Tensor]:
        """|adapted - input| / |input| per replaced level (same size only), for monitoring;
        0-d tensors (no device sync)."""
        out = {}
        for lvl in self.replace_levels:
            a, b = adapted[lvl][:, self.num_aux :], tokens_lvl[lvl][:, self.num_aux :]
            if a.shape == b.shape:
                out[lvl] = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)
        return out


def init_from_source(cf, finest_level: int) -> dict | None:
    """The `init_from` source that loads the adapter weights from the stage-2 run."""
    if not latent_adapter_enabled(cf) or not cf.latent_adapter.get("load", True):
        return None
    res = resolve_latent_adapter(cf)
    return {
        "run_id": res["run_id"],
        "mini_epoch": res["mini_epoch"],
        "include": list(ADAPTER_INCLUDE),
        "rename": {k: v.format(finest=finest_level) for k, v in ADAPTER_RENAME.items()},
    }
