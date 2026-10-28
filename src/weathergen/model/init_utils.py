# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Constructor-equivalent (re-)initialisation of model parameters.

With FSDP the model is built on the `meta` device; to_empty() then allocates
UNINITIALISED storage and every initialisation done in a constructor is lost. The same
holds for modules that are missing from a checkpoint loaded with `assign=True`.
Resetting only nn.Linear / nn.LayerNorm (the previous behaviour) re-creates the default
init but loses the special ones applied by enclosing modules, most importantly the
near-identity N(0, 0.001) init of the forecast engine.

The rules below re-create, for ONE parameter, what the nested constructors do: first the
default of the layer that owns it, then the overrides of its ancestors from the innermost
to the outermost (the order in which the constructors run):

  leaf default                : nn.Linear / nn.LayerNorm / EfficientBilinear
                                -> reset_parameters(); RMSNorm.weight -> 1
  AdaLayerNorm                : last aux Linear -> 0, only if the class zero-inits it
                                (it defines reset_parameters, as in ecmwf develop #2848)
  AdaLayerNormLayer           : modulation Linear -> 0
  Self/CrossAttentionBlock    : every Linear -> xavier, bias 0; the modulation Linear of
                                their AdaLayerNormLayer sub-blocks -> 0 again
  ForecastingEngine           : every Linear in fe_blocks -> N(0, 0.001)
  CascadeBlock (zero_init)    : attention.proj_out.weight -> 0, last MLP Linear -> 0
  _PairOperator ("linear")    : down_proj / up_proj weights -> 0
  TargetPredictionEngine      : pos_embed -> 0

The learnable cell queries (q_cells) are handled by the encoder (reset_q_cells /
initial_q_cells), ModelParams are rebuilt from the config.

Adapted from cristianl/metno-develop-20261027 (ba30bcd, a1f4546) to this branch's module
set.
"""

import logging

import torch
from torch import nn

from weathergen.model.blocks import CrossAttentionBlock, SelfAttentionBlock
from weathergen.model.engines import EfficientBilinear, ForecastingEngine, TargetPredictionEngine
from weathergen.model.latent_cascade import CascadeBlock, _PairOperator
from weathergen.model.norms import AdaLayerNorm, AdaLayerNormLayer, RMSNorm

logger = logging.getLogger(__name__)

# AdaLayerNorm zero-inits its last aux projection once it defines reset_parameters()
# (ecmwf develop 5193404 / #2848); before that merge its init is the plain Linear default
_ADALN_ZERO_INIT = "reset_parameters" in AdaLayerNorm.__dict__


def _zero_(linear: nn.Linear, bias: bool = True) -> None:
    nn.init.zeros_(linear.weight)
    if bias and linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _context_override(anc: nn.Module, owner: nn.Linear, rel_path: str) -> None:
    """Re-apply the init that ancestor `anc` imposes on its descendant Linear `owner`
    (`rel_path` = path of `owner` below `anc`)."""
    if isinstance(anc, AdaLayerNorm):
        if _ADALN_ZERO_INIT and owner is anc.embed_aux[-1]:
            _zero_(owner)
    elif isinstance(anc, AdaLayerNormLayer):
        if owner is anc.adaLN_modulation[-1]:
            _zero_(owner)
    elif isinstance(anc, SelfAttentionBlock | CrossAttentionBlock):
        nn.init.xavier_uniform_(owner.weight)
        if owner.bias is not None:
            nn.init.constant_(owner.bias, 0)
        if anc.with_adanorm:
            for blk_name in ("mhsa_block", "cross_attn_block", "mlp_block"):
                blk = getattr(anc, blk_name, None)
                if isinstance(blk, AdaLayerNormLayer) and owner is blk.adaLN_modulation[-1]:
                    _zero_(owner)
    elif isinstance(anc, ForecastingEngine):
        if rel_path.startswith("fe_blocks."):
            nn.init.normal_(owner.weight, mean=0, std=0.001)
            if owner.bias is not None:
                nn.init.normal_(owner.bias, mean=0, std=0.001)
    elif isinstance(anc, CascadeBlock):
        if anc.zero_init:
            if owner is anc.attention.proj_out:
                nn.init.zeros_(owner.weight)
            elif owner is anc.mlp.layers[-1]:
                _zero_(owner)
    elif isinstance(anc, _PairOperator):
        if anc.kind == "linear" and (owner is anc.down_proj or owner is anc.up_proj):
            nn.init.zeros_(owner.weight)


@torch.no_grad()
def _reinit_parameter(modules: dict[str, nn.Module], key: str, done_owners: set) -> bool:
    """Initialise the parameter with state-dict key `key` as the constructors would.
    Returns False if `key` is not a parameter of a module in `modules`."""
    owner_path, pname = key.rsplit(".", 1) if "." in key else ("", key)
    owner = modules.get(owner_path)
    if owner is None or not isinstance(getattr(owner, pname, None), torch.Tensor):
        return False

    # leaf default (once per owning layer: reset_parameters() covers weight and bias)
    if owner_path not in done_owners:
        done_owners.add(owner_path)
        if isinstance(owner, nn.Linear | nn.LayerNorm | EfficientBilinear):
            owner.reset_parameters()
        elif isinstance(owner, RMSNorm):
            owner.weight.fill_(1.0)

    if isinstance(owner, TargetPredictionEngine) and pname == "pos_embed":
        owner.pos_embed.zero_()
        return True

    # overrides by the enclosing modules, innermost first
    if not isinstance(owner, nn.Linear):
        return True
    parts = owner_path.split(".") if owner_path else []
    for i in range(len(parts) - 1, -1, -1):
        anc = modules.get(".".join(parts[:i]))
        if anc is not None:
            _context_override(anc, owner, ".".join(parts[i:]))
    return True


@torch.no_grad()
def reinit_parameters(root: nn.Module, keys=None) -> None:
    """
    Constructor-equivalent initialisation of the parameters of `root` named in `keys`
    (state-dict keys relative to `root`), or of all of them if `keys` is None. Parameters
    not listed are left untouched. In-place torch.nn.init / fill calls only, so it works
    on FSDP2 DTensor parameters.

    Parameters that the rules do not know (e.g. q_cells) are skipped silently when `keys`
    is None and reported when they were requested explicitly.
    """
    # remove_duplicate=False: a module registered under two names is found under both
    modules = dict(root.named_modules(remove_duplicate=False))
    done_owners: set = set()
    done_params: set = set()
    if keys is None:
        keys = [k for k, _ in root.named_parameters(remove_duplicate=False)]
        explicit = False
    else:
        explicit = True
    for key in sorted(keys):
        owner_path, pname = key.rsplit(".", 1) if "." in key else ("", key)
        param = getattr(modules.get(owner_path), pname, None)
        if param is not None and id(param) in done_params:
            continue  # same tensor already initialised under another name
        if not _reinit_parameter(modules, key, done_owners) and explicit:
            logger.warning(f"reinit_parameters: no parameter for key {key}")
        if param is not None:
            done_params.add(id(param))


def materialize_and_init_missing(model: nn.Module, missing_keys, device) -> None:
    """
    After load_state_dict(..., assign=True) on a meta-device (FSDP) model: allocate the
    parameters listed in `missing_keys` and give them their constructor init, leaving
    every loaded parameter and buffer untouched.

    to_empty() is applied NON-recursively to each module that owns a missing parameter and
    the loaded tensors of that module are put back (a recursive to_empty() on the
    outermost new module, as before, also wiped loaded children). The init itself is done
    on the whole model so that rules depending on an ENCLOSING module are applied, e.g. a
    forecast engine missing from an autoencoder checkpoint gets its near-identity
    N(0, 0.001) init instead of the default Linear init.
    """
    mset = set(missing_keys)
    all_modules = dict(model.named_modules(remove_duplicate=False))
    done_ids: set = set()
    for path in sorted({k.rsplit(".", 1)[0] if "." in k else "" for k in mset}):
        module = all_modules[path]
        if id(module) in done_ids:
            continue
        done_ids.add(id(module))
        prefix = f"{path}." if path else ""
        keep_p = {
            n: p
            for n, p in module.named_parameters(recurse=False)
            if f"{prefix}{n}" not in mset and not p.is_meta
        }
        keep_b = {n: b for n, b in module.named_buffers(recurse=False) if not b.is_meta}
        module.to_empty(device=device, recurse=False)
        for n, p in keep_p.items():
            setattr(module, n, p)
        for n, b in keep_b.items():
            module._buffers[n] = b
    reinit_parameters(model, mset)
