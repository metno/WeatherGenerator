# ruff: noqa: B006

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
import logging

import torch
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.tensor import distribute_tensor

from weathergen.common.config import Config, get_path_model, merge_configs
from weathergen.model.attention import (
    MultiCrossAttentionHeadVarlen,
    MultiCrossAttentionHeadVarlenSlicedQ,
    MultiSelfAttentionHead,
    MultiSelfAttentionHeadLocal,
    MultiSelfAttentionHeadVarlen,
)
from weathergen.model.layers import MLP
from weathergen.model.model import Model, ModelParams, ModelParamsPyramid
from weathergen.model.utils import apply_fct_to_blocks, freeze_weights
from weathergen.utils.distributed import is_root
from weathergen.utils.performance import register_nvtx_hooks
from weathergen.utils.utils import get_dtype

logger = logging.getLogger(__name__)


# same as in config: student_teacher, forecasting, masking
type TrainingMode = str


def init_model_and_shard(
    cf,
    dataset,
    run_id_contd,
    mini_epoch_contd,
    training_mode,
    device,
    with_ddp,
    with_fsdp,
    overrides={},
):
    model_creation_device = "meta" if with_ddp and with_fsdp else "cuda"
    with torch.device(model_creation_device):
        model = get_model(cf, training_mode, dataset, overrides)

    if cf.get("profiling", {}).get("nvtx_annotate", False):
        logger.info("Registering NVTX hooks for model.")
        register_nvtx_hooks(model)

    # freeze request model part
    apply_fct_to_blocks(model, cf.freeze_modules, freeze_weights)

    # TODO: this should be handled in the encoder to be close where q_cells is defined
    if "q_cells" in cf.freeze_modules:
        # q_cells is now a per-level ParameterDict (encoder._q_cells_param_dict);
        # model.encoder.q_cells is a finest-level alias into it. Freeze every level
        # so coarse query banks are frozen too. For a single-level config this is
        # exactly the one (finest) parameter, matching the old single-line freeze.
        _qdict = getattr(model.encoder, "_q_cells_param_dict", None)
        if _qdict is not None:
            for _q in _qdict.values():
                _q.requires_grad = False
        else:
            model.encoder.q_cells.requires_grad = False

    if with_ddp and not with_fsdp:
        # create DDP model if running without FSDP
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            broadcast_buffers=True,
            find_unused_parameters=cf.get("ddp_find_unused_parameters", True),
            gradient_as_bucket_view=True,
            bucket_cap_mb=512,
        )

    elif with_ddp and with_fsdp:
        # with DDP *and() FSDP
        fsdp_kwargs = {
            "mp_policy": (
                MixedPrecisionPolicy(
                    param_dtype=get_dtype(cf.mixed_precision_dtype),
                    reduce_dtype=torch.float32,
                )
                if cf.with_mixed_precision
                else None
            ),
        }
        modules_to_shard = (
            MLP,
            MultiSelfAttentionHeadLocal,
            MultiSelfAttentionHead,
            MultiCrossAttentionHeadVarlen,
            MultiCrossAttentionHeadVarlenSlicedQ,
            MultiSelfAttentionHeadVarlen,
        )

        def _shard_leaf_modules(root, **kwargs):
            """Shard every leaf in `root` whose type is in modules_to_shard."""
            for module in root.modules():
                if isinstance(module, modules_to_shard):
                    fully_shard(module, **kwargs)

        # --- local + local->global adapter (SHARED across levels: single set of
        # weights, unchanged from the single-level baseline). ---------------------
        # NOTE: when share_local_assimilation=False these become per-level dicts
        # (encoder._ae_local_per_level / _ae_local_global_per_level). The aliases
        # below still point at the finest level, so the non-shared branch is not
        # yet fully sharded -- tracked as a follow-up once the shared gate passes.
        _shard_leaf_modules(model.encoder.ae_local_engine.ae_local_blocks, **fsdp_kwargs)
        _shard_leaf_modules(model.encoder.ae_local_global_engine.ae_adapter, **fsdp_kwargs)

        # --- global assimilation: one GlobalAssimilationEngine PER LEVEL. --------
        # ae_global_engine is now a finest-level alias into _ae_global_per_level,
        # so sharding only the alias would leave coarser levels unsharded. Loop the
        # dict. For a single-level config this dict has exactly one (finest) entry,
        # so the set of sharded modules is identical to the old alias-only code
        # -> the single-level FSDP gate is byte-identical.
        for _lvl, _eng in model.encoder._ae_global_per_level.items():
            _shard_leaf_modules(_eng.ae_global_blocks, **fsdp_kwargs)

        # --- query-aggregation engines are also per-level. Each holds real
        # attention params (QueryAggregationEngine.ae_aggregation_blocks:
        # MultiSelfAttentionHeadVarlen / ...Local). In the single-level baseline
        # these were NOT wrapped here -- the trailing fully_shard(model) sweeps
        # them into the ROOT group. That still happens, so the gate is unchanged.
        # Follow-up for multi-level: root-group params don't reshard across the
        # fwd/bwd boundary, so wrapping each level's ae_aggregation_blocks
        # explicitly (like ae_global above) would cut peak memory. Left out now to
        # keep the single-level shard set identical. -------------------------------

        # --- forecast engine: one ForecastingEngine PER LEVEL (or an IdentityEngine
        # when fe_num_blocks == 0, which has no fe_blocks). Loop the dict when it
        # exists; fall back to the alias for the identity/no-forecast case. --------
        # reshard_after_forward=False keeps FE params unsharded during the
        # multi-step rollout loop (needed for the pushforward trick).
        _fe_per_level = getattr(model, "_forecast_per_level", None)
        if _fe_per_level is not None:
            for _lvl, _fe in _fe_per_level.items():
                _shard_leaf_modules(_fe.fe_blocks, reshard_after_forward=False, **fsdp_kwargs)
        elif hasattr(model.forecast_engine, "fe_blocks"):
            _shard_leaf_modules(
                model.forecast_engine.fe_blocks, reshard_after_forward=False, **fsdp_kwargs
            )

        # --- latent cascades (encoder.latent_cascade + model.forecast_cascade).
        # These use plain nn.Linear / a custom _CrossAttn, none of which are in
        # modules_to_shard, so the leaf-shard loop is a no-op on them today. With
        # operator="none" (the single-level default) they are parameter-inert
        # anyway. Left unsharded on purpose to keep the gate identical; when the
        # linear/attention operators are enabled at scale, shard each cascade as
        # its own FSDP unit here. --------------------------------------------------

        for module in model.latent_heads.modules():
            if isinstance(module, modules_to_shard):
                fully_shard(module, **fsdp_kwargs)

        full_precision_fsdp_kwargs = {
            "mp_policy": (
                MixedPrecisionPolicy(
                    param_dtype=torch.float32,
                    reduce_dtype=torch.float32,
                )
                if cf.with_mixed_precision
                else None
            ),
        }

        for module in model.target_token_engines.modules():
            if isinstance(module, modules_to_shard):
                fully_shard(module, **full_precision_fsdp_kwargs)

    if with_ddp and with_fsdp:
        fully_shard(model)
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            assert tensor.device == torch.device("meta")

        # For reasons we do not yet fully understand, when using train continue in some
        # instances, FSDP2 does not register the forward_channels and forward_columns
        # functions in the embedding engine as forward functions. Thus, yielding a crash
        # because the input tensors are not converted to DTensors. This seems to primarily
        # occur during validation.
        for embed in model.encoder.embed_engine.embeds.values():
            torch.distributed.fsdp.register_fsdp_forward_method(embed, "forward")

    # complete initalization and load model if inference/continuing a run
    if run_id_contd is not None:
        if is_root():
            logger.info(f"Continuing run with id={run_id_contd} at mini_epoch {mini_epoch_contd}.")
        model = load_model(cf, model, device, run_id_contd, mini_epoch_contd)
    elif cf.get("load_chkpt", {}).get("run_id", None):
        run_id = cf.load_chkpt.run_id
        mini_epoch = cf.load_chkpt.get("mini_epoch", -1)
        if is_root():
            logger.info(f"Loading checkpoint from id={run_id} at mini_epoch {mini_epoch}.")
        model = load_model(cf, model, device, run_id, mini_epoch)
    else:
        if with_ddp and with_fsdp:
            model.to_empty(device="cuda")
            if with_fsdp:
                model.reset_parameters()

    # model params
    model_params = ModelParamsPyramid(cf).create(cf)
    model_params.reset_parameters(cf)
    model_params = model_params.to(f"cuda:{cf.local_rank}")

    # TEMP diagnostic -- remove after
#    def _nan_hook_named(name, mod, inp, out):
#        def _bad(x):
#            return isinstance(x, torch.Tensor) and x.is_floating_point() and torch.isnan(x).any()
#        outs = out if isinstance(out, tuple) else (out,)
#        ins = [i for i in inp if isinstance(i, torch.Tensor)]
#        if any(_bad(o) for o in outs) and not any(_bad(i) for i in ins):
#            print(f"FIRST NaN CREATED IN: {name} ({mod.__class__.__name__})", flush=True)
#            for _i, _t in enumerate(ins):
#                v = _t.float().var(dim=-1).min().item() if _t.numel() else float("nan")
#                print(f"   input[{_i}] shape={tuple(_t.shape)} "
#                      f"min_row_var={v:.3e} absmax={_t.abs().max().item():.3e}", flush=True)
#            w = getattr(mod, "weight", None)                                          # ADD
#            if w is not None:                                                         # ADD
#                wl = w.to_local() if hasattr(w, "to_local") else w                    # ADD
#                print(f"   weight: shard_nan={torch.isnan(wl).any().item()} "         # ADD
#                      f"shard_absmax={wl.abs().max().item():.3e}", flush=True)        # ADD
#                print(f"   ADDR-3 CRASH: addr={wl.data_ptr():#x} shape={tuple(wl.shape)}", flush=True)
#                print(f"   dtype={wl.dtype} is_dtensor={hasattr(w, 'to_local')}", flush=True)

#    for _mn, _m in model.named_modules():
#        _m.register_forward_hook(
#            lambda mod, inp, out, _nm=_mn: _nan_hook_named(_nm, mod, inp, out)
#        )

    # Repair LatentCascade geometry buffers after meta-device init (FSDP path).
    # The cascade registers its parent/child index maps (poc_/cop_/cv_) as
    # NON-persistent buffers, so they are (a) created on `meta` when the model is
    # built on meta, and (b) skipped by load_state_dict -- neither to_empty nor
    # reset_parameters (which only touches Linear/LayerNorm) repopulates them.
    # For a single-level config the cascade has no pairs and no buffers, so this
    # loop is a no-op and the gate is unaffected. With >=2 levels enabled it
    # rebuilds the maps on-device from the pyramid so the geometry is correct.
    if with_ddp and with_fsdp:
        for _casc in (
            getattr(model.encoder, "latent_cascade", None),
            getattr(model, "forecast_cascade", None),
        ):
            if _casc is None or getattr(_casc, "is_inert", lambda: True)():
                continue
            _pyr = _casc.pyramid
            _dev = torch.device(f"cuda:{cf.local_rank}")
            for lc, lf in _casc.pairs:
                key = f"{lc}_{lf}"
                poc = torch.as_tensor(_pyr.parent_of_child(lc, lf), dtype=torch.long, device=_dev)
                cop = torch.as_tensor(_pyr.child_of_parent(lc, lf), dtype=torch.long, device=_dev)
                cv = torch.as_tensor(_pyr.child_valid(lc, lf), dtype=torch.bool, device=_dev)
                # overwrite in place (buffers already exist as attributes)
                setattr(_casc, f"poc_{key}", poc)
                setattr(_casc, f"cop_{key}", cop)
                setattr(_casc, f"cv_{key}", cv)

#    for _n, _p in model.named_parameters():
#        if _p is not None and torch.isnan(_p).any():
#            print(f"UNINIT PARAM: {_n} shape={tuple(_p.shape)}", flush=True)

#    if with_ddp and with_fsdp:
#        _adapter = model.encoder.ae_local_global_engine.ae_adapter
#        _checked = 0
#        _nan_found = 0
#        for _n, _p in _adapter.named_parameters():
#            try:
#                _full = _p.full_tensor() if hasattr(_p, "full_tensor") else _p
#            except Exception as _e:
#                if is_root():
#                    print(f"SHARD CHECK ERROR on {_n}: {_e}", flush=True)
#                continue
#            _checked += 1
#            if torch.isnan(_full).any():
#                _nan_found += 1
#                if is_root():
#                    print(f"UNINIT SHARD (gathered NaN): ae_adapter.{_n} shape={tuple(_full.shape)}", flush=True)
#        if is_root():
#            print(f"SHARD CHECK DONE: checked={_checked} nan={_nan_found}", flush=True)
#    if with_ddp and with_fsdp:
#        _gpl = model.encoder._ae_global_per_level
#        if is_root():
#            print(f"GLOBAL8 KEYS: {list(_gpl.keys())}", flush=True)
#        _key = "8" if "8" in _gpl else (8 if 8 in _gpl else list(_gpl.keys())[-1])
#        _eng = _gpl[_key].ae_global_blocks
##        _eng = model.encoder._ae_global_per_level["8"].ae_global_blocks
#        _checked = _nan = 0
#        for _n, _p in _eng.named_parameters():
#            _full = _p.full_tensor() if hasattr(_p, "full_tensor") else _p
#            _checked += 1
#            if "4.proj_heads_q" in _n:
#                _l = _p.to_local() if hasattr(_p, "to_local") else _p
#                print(f"ADDR-1 INIT: {_n} addr={_l.data_ptr():#x} "
#                      f"nan={torch.isnan(_l).any().item()} absmax={_l.abs().max().item():.3e}", flush=True)
#            if torch.isnan(_full).any():
#                _nan += 1
#                if is_root():
#                    print(f"GLOBAL8 NaN SHARD: {_n} shape={tuple(_full.shape)}", flush=True)
#        if is_root():
#            print(f"GLOBAL8 CHECK: checked={_checked} nan={_nan}", flush=True)

    _bad = [n for n, p in model.named_parameters() if p is not None and torch.isnan(p).any()]
    assert not _bad, f"Uninitialized/NaN parameters after init: {_bad[:5]}"

    return model, model_params


def load_model(cf, model, device, run_id: str, mini_epoch=-1):
    """Loads model state from checkpoint and checks for missing and unused keys.
    Args:
        run_id : model_id of the trained model
        mini_epoch : The mini_epoch to load. Default (-1) is the latest mini_epoch
    """

    path_run = get_path_model(run_id=run_id)
    mini_epoch_id = (
        f"chkpt{mini_epoch:05d}" if mini_epoch != -1 and mini_epoch is not None else "latest"
    )
    filename = f"{run_id}_{mini_epoch_id}.chkpt"

    params = torch.load(
        path_run / filename, map_location=torch.device("cpu"), mmap=True, weights_only=True
    )

    is_model_sharded = cf.with_ddp and cf.with_fsdp
    if is_model_sharded:
        meta_sharded_sd = model.state_dict()
        maybe_sharded_sd = {}
        for param_name, full_tensor in params.items():
            sharded_meta_param = meta_sharded_sd.get(param_name)
            if sharded_meta_param is None:
                logger.warning(f"Parameter {param_name} from checkpoint not found in model.")
                continue
            sharded_tensor = distribute_tensor(
                full_tensor,
                sharded_meta_param.device_mesh,
                sharded_meta_param.placements,
            )
            # maybe_sharded_sd[param_name.replace("module.", "")] = nn.Parameter(sharded_tensor)
            maybe_sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)
        # choose `assign=True` for sharded model since we cannot call `copy_` on meta tensor
        mkeys, ukeys = model.load_state_dict(maybe_sharded_sd, strict=False, assign=True)

        # new network parts (e.g. for fine-tuning)
        if mkeys:
            # Get the unique parent modules for the missing parameters
            new_modules_to_init = {key.rsplit(".", 1)[0] for key in mkeys}

            # Find the highest-level "root" new modules to avoid redundant initializations
            root_new_modules = set()
            for path in sorted(list(new_modules_to_init)):
                if not any(path.startswith(root + ".") for root in root_new_modules):
                    root_new_modules.add(path)

            # Get all modules for quick lookup and initialize the new ones
            all_modules = dict(model.named_modules())
            for path in root_new_modules:
                if is_root():
                    logger.info(f"Initializing new module not found in checkpoint: {path}")
                module_to_init = all_modules[path]
                module_to_init.to_empty(device="cuda")
                module_to_init.reset_parameters()

    else:
        # fix mismatch between state_dict keys that can occur between interactive/non-interactive
        model_has_prefix_module = list(model.state_dict().keys())[0].split(".")[0] == "module"
        params_has_prefix_module = list(params.keys())[0].split(".")[0] == "module"
        if model_has_prefix_module and not params_has_prefix_module:
            # add "module." prefix
            params_temp = {}
            for k in params.keys():
                params_temp["module." + k] = params[k]
            params = params_temp
        elif not model_has_prefix_module and params_has_prefix_module:
            # remove "module." prefix
            params_temp = {}
            for k in params.keys():
                params_temp[k.replace("module.", "")] = params[k]
            params = params_temp
        # load checkpoint
        mkeys, ukeys = model.load_state_dict(params, strict=False)
        model = model.to(device)

    # warn about difference in checkpoint and model
    if len(mkeys) == 0 and len(ukeys) == 0:
        logger.info(f"Checkpoint {filename} loaded successfully with all weights matching.")
    if len(mkeys) > 0:
        logger.warning(f"Missing keys when loading model: {mkeys}")
    if len(ukeys) > 0:
        logger.warning(f"Unused keys when loading model: {ukeys}")

    return model


def get_model(cf: Config, training_mode: TrainingMode, dataset, overrides):
    """
    Create model

    cf :
    training_mode :
    dataset :
    """

    # TODO: how to avoid the dependence on dataset
    sources_size = dataset.get_sources_size()
    targets_num_channels = dataset.get_targets_num_channels()
    targets_coords_size = dataset.get_targets_coords_size()

    cf_with_overrides = merge_configs(cf, overrides)
    return Model(
        cf_with_overrides, sources_size, targets_num_channels, targets_coords_size
    ).create()
