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
import re

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
from weathergen.model.init_utils import materialize_and_init_missing
from weathergen.model.latent_adapter import init_from_source as adapter_init_from_source
from weathergen.model.latent_adapter import latent_adapter_enabled
from weathergen.model.layers import MLP
from weathergen.model.model import Model, build_model_params
from weathergen.model.utils import apply_fct_to_blocks, freeze_weights, unfreeze_weights
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

    # freeze request model part (freeze_modules), minus trainable_modules
    apply_freezing(cf, model)

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

        for module in model.encoder.ae_local_engine.ae_local_blocks.modules():
            if isinstance(module, modules_to_shard):
                fully_shard(module, **fsdp_kwargs)

        for module in model.encoder.ae_local_global_engine.ae_adapter.modules():
            if isinstance(module, modules_to_shard):
                fully_shard(module, **fsdp_kwargs)

        for module in model.encoder.ae_global_engine.ae_global_blocks.modules():
            if isinstance(module, modules_to_shard):
                fully_shard(module, **fsdp_kwargs)

        for module in model.forecast_engine.fe_blocks.modules():
            if isinstance(module, modules_to_shard):
                # reshard_after_forward=False keeps FE parameters unsharded
                # during the multi-step rollout loop.
                # Needed for pushforward trick.
                fully_shard(module, reshard_after_forward=False, **fsdp_kwargs)

        # multi-resolution latent: the same sharding for the engines of the coarser levels
        # (the finest level uses the modules above). Query aggregation engines are swept
        # into the root group by fully_shard(model), as for one level.
        if model.encoder.multi_level:
            per_level_blocks = [
                e.ae_global_blocks for e in model.encoder._ae_global_per_level.values()
            ]
            if not model.encoder.share_local_assimilation:
                per_level_blocks += [
                    e.ae_local_blocks for e in model.encoder._ae_local_per_level.values()
                ]
            # own adapters: unshared, or a level with another latent size (dim_embed)
            if model.encoder._ae_local_global_per_level is not None:
                per_level_blocks += [
                    e.ae_adapter for e in model.encoder._ae_local_global_per_level.values()
                ]
            for blocks in per_level_blocks:
                for module in blocks.modules():
                    if isinstance(module, modules_to_shard):
                        fully_shard(module, **fsdp_kwargs)
            for engine in (model._forecast_per_level or {}).values():
                for module in engine.fe_blocks.modules():
                    if isinstance(module, modules_to_shard):
                        fully_shard(module, reshard_after_forward=False, **fsdp_kwargs)
            # attention blocks of the latent cascades (cross-attention + MLP)
            for cascade in model.latent_cascades():
                for module in cascade.modules():
                    if isinstance(module, modules_to_shard):
                        fully_shard(module, **fsdp_kwargs)

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
        assert not cf.get("init_from", None), "Use either load_chkpt or init_from, not both."
        run_id = cf.load_chkpt.run_id
        mini_epoch = cf.load_chkpt.get("mini_epoch", -1)
        if is_root():
            logger.info(f"Loading checkpoint from id={run_id} at mini_epoch {mini_epoch}.")
        model = load_model(cf, model, device, run_id, mini_epoch)
    elif cf.get("init_from", None) or latent_adapter_enabled(cf):
        # warm start from parts of several runs (e.g. encoder of a JEPA run + decoder of an
        # autoencoder run, + the kept predictor of the latent adapter); everything not
        # loaded keeps its fresh initialisation
        model = load_init_from(cf, model, device)
    else:
        if with_ddp and with_fsdp:
            model.to_empty(device="cuda")
            if with_fsdp:
                model.reset_parameters()

    # the latent cascades keep their parent/child index maps in non-persistent buffers: with
    # meta-device initialisation (FSDP) they are neither in the checkpoint nor rebuilt by
    # to_empty / reset_parameters, so rebuild them from the pyramid on the device
    if with_ddp and with_fsdp:
        for cascade in model.latent_cascades():
            cascade.rebuild_index_buffers(torch.device(f"cuda:{cf.local_rank}"))
        # same for the cell centres of the multi-level JEPA predictor
        model.rebuild_ssl_buffers(torch.device(f"cuda:{cf.local_rank}"))

    # FSDP loads with assign=True, which replaces the parameters (requires_grad=True):
    # freeze again after loading
    if with_ddp and with_fsdp:
        apply_freezing(cf, model, log=False)
    if is_root():
        log_trainable_parameters(model)

    # model params (one set per latent level for a multi-resolution latent)
    model_params = build_model_params(cf).create(cf)
    model_params.reset_parameters(cf)
    model_params = model_params.to(f"cuda:{cf.local_rank}")

    return model, model_params


def _as_regex(patterns) -> str:
    """A regex string, or a list of them (alternatives); "" matches nothing."""
    if patterns is None:
        return ""
    if isinstance(patterns, str):
        return patterns
    patterns = [str(p) for p in patterns]
    return "|".join(f"(?:{p})" for p in patterns) if patterns else ""


def apply_freezing(cf, model, log: bool = True) -> None:
    """
    Freeze the modules whose name fully matches `freeze_modules` (a regex or a list of
    regexes), then make the modules matching `trainable_modules` trainable again (exceptions,
    e.g. freeze `encoder\\..*` but train `encoder\\.latent_cascade.*`). Module names are
    those of model.named_modules() (or a module's own `name` attribute, e.g. "JEPA-head").
    """
    base = getattr(model, "module", model)
    freeze = _as_regex(cf.get("freeze_modules", ""))
    trainable = _as_regex(cf.get("trainable_modules", ""))
    if freeze:
        apply_fct_to_blocks(base, freeze, freeze_weights if log else _freeze_quiet)
        # TODO: this should be handled in the encoder to be close where q_cells is defined
        if "q_cells" in freeze:
            # all latent levels (only model.encoder.q_cells for a single latent level)
            for q_cells in base.encoder.q_cells_all_levels():
                q_cells.requires_grad = False
    if trainable:
        apply_fct_to_blocks(base, trainable, unfreeze_weights if log else _unfreeze_quiet)
        if "q_cells" in trainable:
            for q_cells in base.encoder.q_cells_all_levels():
                q_cells.requires_grad = True


def _freeze_quiet(block):
    for p in block.parameters():
        p.requires_grad = False


def _unfreeze_quiet(block):
    for p in block.parameters():
        p.requires_grad = True


def log_trainable_parameters(model) -> None:
    """Trainable / frozen parameter counts per top-level module."""
    base = getattr(model, "module", model)
    counts: dict[str, list[int]] = {}
    for name, p in base.named_parameters():
        top = name.split(".")[0]
        c = counts.setdefault(top, [0, 0])
        c[0 if p.requires_grad else 1] += p.numel()
    if not any(c[1] for c in counts.values()):
        return
    lines = [f"  {k}: trainable {c[0]:,}, frozen {c[1]:,}" for k, c in sorted(counts.items())]
    logger.info("Parameters per module:\n" + "\n".join(lines))


def _checkpoint_file(cf, run_id: str, mini_epoch=-1):
    path_run = get_path_model(cf, run_id=run_id)
    mini_epoch_id = (
        f"chkpt{mini_epoch:05d}" if mini_epoch != -1 and mini_epoch is not None else "latest"
    )
    return path_run / f"{run_id}_{mini_epoch_id}.chkpt"


# what load_init_from loaded, for the run monitor (train/jepa_monitor.py):
# {"loaded": {parameter name: run_id}}; None when init_from is not used
INIT_FROM_REPORT: dict | None = None


_DECODER_PREFIXES = ("embed_target_coords.", "target_token_engines.", "pred_heads.")


def _check_increment_baseline(cf, run_id, mini_epoch, keys) -> None:
    """
    Warn when a decoder is taken from a run whose stream had a different `increment_baseline`
    setting than the stream in this run (datasets/increment_baseline.py): a decoder trained on
    increments w.r.t. the baseline must be run with the baseline, and vice versa.
    """
    dec_streams = {k.split(".")[1] for k in keys if k.startswith(_DECODER_PREFIXES)}
    if not dec_streams or not is_root():
        return
    try:
        from weathergen.common.config import load_run_config

        src_cf = load_run_config(run_id, mini_epoch, model_path=cf.get("model_path", None))
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"init_from: config of {run_id} not readable ({e!r}); "
            "increment_baseline consistency of its decoders not checked"
        )
        return
    for s in sorted(dec_streams):
        src = ((src_cf.get("streams", None) or {}).get(s, None) or {}).get("increment_baseline")
        own = ((cf.get("streams", None) or {}).get(s, None) or {}).get("increment_baseline")
        if (src is None) != (own is None):
            logger.warning(
                f"init_from: the {s} decoder of {run_id} was trained "
                f"{'with' if src is not None else 'without'} increment_baseline, but stream {s} "
                f"is configured {'with' if own is not None else 'without'} it in this run: the "
                "decoder output will be interpreted differently"
            )


def load_init_from(cf, model, device):
    """
    Warm start from parts of several runs (config `init_from`, a list of sources):

        init_from:
          - run_id: <JEPA run>         # e.g. stage 2: student encoders + latent cascade
            mini_epoch: -1
            include: ["encoder\\..*"]   # regexes on parameter names (re.match), default all
            exclude: ["encoder\\.embed_engine\\.embeds\\.CARRA2\\..*"]
          - run_id: <autoencoder run>  # e.g. stage 1: the CARRA2 decoder
            include: ["embed_target_coords\\.CARRA2\\..*", "target_token_engines\\.CARRA2\\..*",
                      "pred_heads\\.CARRA2\\..*"]
            rename: {"^old_prefix": "new_prefix"}   # optional re.sub on the names

    Later sources override earlier ones for the same parameter. Parameters found in no
    source keep their fresh initialisation (FSDP: constructor init, as for load_chkpt).
    A selected parameter whose shape differs from the model's is an error.
    """
    global INIT_FROM_REPORT
    merged: dict = {}
    origin: dict[str, str] = {}
    base = getattr(model, "module", model)
    sources = list(cf.get("init_from", None) or [])
    adapter_src = adapter_init_from_source(cf, getattr(base, "healpix_level", None))
    if adapter_src is not None:
        sources.append(adapter_src)
    for src in sources:
        run_id = src["run_id"]
        mini_epoch = src.get("mini_epoch", -1)
        include = _as_regex(src.get("include", None)) or ".*"
        exclude = _as_regex(src.get("exclude", None))
        rename = dict(src.get("rename", None) or {})
        filename = _checkpoint_file(cf, run_id, mini_epoch)
        params = torch.load(
            filename, map_location=torch.device("cpu"), mmap=True, weights_only=True
        )
        n = 0
        for k, v in params.items():
            k = k.removeprefix("module.")
            if re.match(include, k) is None or (exclude and re.match(exclude, k) is not None):
                continue
            for pat, repl in rename.items():
                k = re.sub(pat, repl, k)
            if k in merged and is_root():
                logger.info(f"init_from: {k} from {run_id} overrides an earlier source")
            merged[k] = v
            origin[k] = str(run_id)
            n += 1
        assert n > 0, f"init_from: no parameter of run {run_id} matches include/exclude"
        _check_increment_baseline(
            cf, run_id, mini_epoch, [k for k, r in origin.items() if r == str(run_id)]
        )
        if is_root():
            logger.info(f"init_from: {n} tensors selected from {filename.name}")

    model_sd = base.state_dict()
    unknown = [k for k in merged if k not in model_sd]
    mismatch = [
        f"{k}: checkpoint {tuple(v.shape)} vs model {tuple(model_sd[k].shape)}"
        for k, v in merged.items()
        if k in model_sd and tuple(v.shape) != tuple(model_sd[k].shape)
    ]
    assert not mismatch, (
        "init_from: shape mismatch (e.g. a decoder needs the latent size of the run it was "
        "trained with):\n  " + "\n  ".join(mismatch[:20])
    )
    if unknown and is_root():
        logger.warning(
            f"init_from: {len(unknown)} selected tensors not in the model: {unknown[:20]}"
        )
    merged = {k: v for k, v in merged.items() if k in model_sd}
    if adapter_src is not None:
        # the kept predictor must come completely from the stage-2 run
        missing = [
            k
            for k, _ in base.named_parameters()
            if k.startswith("latent_adapter.") and k not in merged
        ]
        assert not missing, (
            f"latent_adapter: parameters not found in run {adapter_src['run_id']}: {missing[:10]}"
        )
    INIT_FROM_REPORT = {"loaded": {k: origin[k] for k in merged}}
    return load_params_into_model(cf, model, device, merged, "init_from")


def load_model(cf, model, device, run_id: str, mini_epoch=-1):
    """Loads model state from checkpoint and checks for missing and unused keys.
    Args:
        run_id : model_id of the trained model
        mini_epoch : The mini_epoch to load. Default (-1) is the latest mini_epoch
    """

    path = _checkpoint_file(cf, run_id, mini_epoch)
    params = torch.load(path, map_location=torch.device("cpu"), mmap=True, weights_only=True)
    return load_params_into_model(cf, model, device, params, path.name)


def load_params_into_model(cf, model, device, params: dict, filename: str):
    """Load a state dict (full tensors) into a (possibly FSDP-sharded) model."""
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
        # learnable query banks missing from the checkpoint (e.g. a latent level added for
        # fine-tuning): they are plain Parameters, not covered by the module re-init below,
        # so load a fresh initialisation through the same sharded path
        for param_name, sharded_meta_param in meta_sharded_sd.items():
            if param_name in maybe_sharded_sd:
                continue
            q_init = model.encoder.initial_q_cells(param_name)
            if q_init is None:
                continue
            if is_root():
                logger.info(f"Initializing {param_name} (not found in checkpoint).")
            maybe_sharded_sd[param_name] = torch.nn.Parameter(
                distribute_tensor(
                    q_init.to(sharded_meta_param.dtype),
                    sharded_meta_param.device_mesh,
                    sharded_meta_param.placements,
                )
            )
        # choose `assign=True` for sharded model since we cannot call `copy_` on meta tensor
        mkeys, ukeys = model.load_state_dict(maybe_sharded_sd, strict=False, assign=True)

        # new network parts (e.g. for fine-tuning)
        if mkeys:
            # Get the unique parent modules for the missing parameters
            new_modules_to_init = {key.rsplit(".", 1)[0] for key in mkeys}

            # Find the highest-level "root" new modules (for logging)
            root_new_modules = set()
            for path in sorted(list(new_modules_to_init)):
                if not any(path.startswith(root + ".") for root in root_new_modules):
                    root_new_modules.add(path)

            if is_root():
                for path in sorted(root_new_modules):
                    logger.info(f"Initializing new module not found in checkpoint: {path}")
            # allocate only the missing parameters and give them their constructor init
            # (incl. the overrides of enclosing modules, e.g. the forecast engine's
            # near-identity init when continuing from an autoencoder checkpoint)
            materialize_and_init_missing(model, mkeys, device="cuda")

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
