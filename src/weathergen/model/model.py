# ruff: noqa: T201
# (C) Copyright 2025 WeatherGenerator contributors.

#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools
import logging
import math
import typing
import warnings

import astropy_healpix as hp
import astropy_healpix.healpy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import Config
from weathergen.datasets.batch import ModelBatch
from weathergen.datasets.utils import healpix_verts_rots, r3tos2
from weathergen.model.encoder import EncoderModule
from weathergen.model.engines import (
    BilinearDecoder,
    EnsPredictionHead,
    ForecastingEngine,
    IdentityEngine,
    LatentPredictionHeadIdentity,
    LatentPredictionHeadMLP,
    LatentPredictionHeadTransformer,
    LatentState,
    TargetPredictionEngine,
    TargetPredictionEngineClassic,
)
from weathergen.model.layers import MLP, NamedLinear
from weathergen.model.utils import get_num_parameters
from weathergen.utils.distributed import is_root
from weathergen.utils.utils import get_dtype, is_stream_forcing
from weathergen.datasets.domain import Domain
from weathergen.datasets.domain_pyramid import build_domain_pyramid

logger = logging.getLogger(__name__)

type StreamName = str


class ModelOutput:
    """
    Representation of model output
    """

    physical: list[dict[StreamName, torch.Tensor]]
    latent: list[dict[str, torch.Tensor | LatentState]]

    def __init__(self, len_output: int) -> None:
        self.physical = [{} for _ in range(len_output)]
        self.latent = [{} for _ in range(len_output)]

    def add_physical_prediction(
        self, fstep: int, stream_name: StreamName, pred: torch.Tensor
    ) -> None:
        self.physical[fstep][stream_name] = pred

    def add_latent_prediction(self, fstep: int, latent_name: str, pred: torch.Tensor) -> None:
        self.latent[fstep][latent_name] = pred

    def get_physical_prediction(
        self, fstep: int, stream_name: StreamName | None = None, sample_idx: int | None = None
    ):
        pred = self.physical[fstep]
        if stream_name is not None:
            pred = pred.get(stream_name, None)
            if sample_idx is not None:
                assert sample_idx < len(pred), "Invalid sample index."
                pred = pred[sample_idx]
        return pred

    def get_latent_prediction(self, fstep: int):
        return self.latent[fstep]


class ModelParams(torch.nn.Module):
    """Creation of query and embedding parameters of the model."""

    def __init__(self, cf) -> None:
        super(ModelParams, self).__init__()

        self.cf = cf

        self.healpix_level = cf.healpix_level
        # Phase 2: build the (single-level) latent domain via the pyramid helper.
        # With no `latent_levels` in config this is bit-identical to
        # Domain.from_config(cf); it introduces the seam that later steps use to
        # add coarse/fine levels without re-plumbing every call site.
        self.domain_pyramid = build_domain_pyramid(cf)
        self.domain = self.domain_pyramid.domain(self.domain_pyramid.finest)
        self.num_healpix_cells = len(self.domain)
        self.dtype = get_dtype(cf.attention_dtype)

        # Positional embeddings
        self.max_tokens_local_per_cell = cf.get("ae_local_max_tokens_per_cell", 64)
        self.pe_embed = torch.nn.Parameter(
            torch.zeros(self.max_tokens_local_per_cell, cf.ae_local_dim_embed, dtype=self.dtype),
            requires_grad=False,
        )

        pe = torch.zeros(
            self.num_healpix_cells,
            cf.ae_local_num_queries,
            cf.ae_global_dim_embed,
            dtype=self.dtype,
        )
        self.pe_global = torch.nn.Parameter(pe, requires_grad=False)

        # RoPE coordinates
        self.rope_2D = cf.get("rope_2D", False)
        if self.rope_2D:
            self.num_extra_tokens = cf.num_register_tokens + cf.num_class_tokens
            total_tokens = (
                self.num_healpix_cells + self.num_extra_tokens
            ) * cf.ae_local_num_queries
            self.register_buffer(
                "rope_coords",
                torch.zeros(
                    1,
                    total_tokens,
                    2,
                    dtype=self.dtype,
                ),
            )
            self.register_buffer(
                "rope_cell_coords",
                torch.zeros(
                    self.num_healpix_cells,
                    2,
                    dtype=self.dtype,
                ),
            )
        else:
            self.rope_coords = None
            self.rope_cell_coords = None

        # HEALPix neighbours, in compact (domain-local) indexing.
        # Out-of-domain neighbours fall back to self, exactly as healpix-corner
        # cells with no 8th neighbour already do.
        temp = self.domain.neighbours_compact()
        self.hp_nbours = torch.nn.Parameter(
            torch.empty((temp.shape[0], (temp.shape[1] + 1)), dtype=torch.int32),
            requires_grad=False,
        )

        self.q_cells_lens = torch.nn.Parameter(
            torch.ones(self.num_healpix_cells + 1, dtype=torch.int32), requires_grad=False
        )
        self.q_cells_lens.data[0] = 0

    def create(self, cf: Config) -> "ModelParams":
        self.reset_parameters(cf)
        return self

    def reset_parameters(self, cf: Config) -> "ModelParams":
        """Creates positional embedding for each grid point for each stream used after stream
        embedding, positional embedding for all stream assimilated cell-level local embedding,
        initializing queries for local-to-global adapters, HEALPix neighbourhood based parameter
        initializing for target prediction.

        Sinusoidal positional encoding: Harmonic positional encoding based upon sine and cosine for
            both per stream after stream embedding and per cell level for local assimilation.

        HEALPix neighbourhood structure: Determine the neighbors for each cell and initialize each
            with its own cell number as well as the cell numbers of its neighbors. If a cell has
            fewer than eight neighbors, use its own cell number to fill the remaining slots.

        Query len based parameter creation: Calculate parameters for the calculated token length at
            each cell after local assimilation.

        Args:
            cf : Configuration
        """

        # positional encodings

        dim_embed = cf.ae_local_dim_embed
        token_idx_bias = 16
        freq_bias = 8
        self.pe_embed.data.fill_(0.0)
        position = torch.arange(
            token_idx_bias,
            token_idx_bias + self.max_tokens_local_per_cell,
            device=self.pe_embed.device,
        ).unsqueeze(1)
        div = torch.exp(
            torch.arange(freq_bias, freq_bias + dim_embed, 2, device=self.pe_embed.device)
            * -(math.log(self.max_tokens_local_per_cell) / dim_embed),
        )
        self.pe_embed.data[:, 0::2] = torch.sin(position * div[: self.pe_embed[:, 0::2].shape[1]])
        self.pe_embed.data[:, 1::2] = torch.cos(position * div[: self.pe_embed[:, 1::2].shape[1]])

        dim_embed = cf.ae_global_dim_embed

        if self.rope_2D:
            # Precompute per-cell center coordinates (lat, lon in radians) for 2D RoPE.
            # Shape: (num_healpix_cells, ae_local_num_queries, 2)
            verts, _ = healpix_verts_rots(self.healpix_level, 0.5, 0.5)
#            print(f"A verts full: {tuple(verts.shape)}", flush=True)
            # TEMP -- remove after
#            ac = self.domain.active_cells
#            print(f"DOMAIN IN MODELPARAMS: n={len(ac)} range=[{ac.min()},{ac.max()}] "
#                  f"first5={ac[:5].tolist()} is_global={self.domain.is_global}", flush=True)
#            import weathergen.datasets.utils as _u
#            print(f"UTILS FROM: {_u.__file__}", flush=True)
            verts = verts[torch.from_numpy(self.domain.active_cells)]
#            print(f"B verts sub:  {tuple(verts.shape)}", flush=True)
            coords = r3tos2(verts.to(self.rope_coords.device)).to(self.rope_coords.dtype)
            coords[:, 1] = torch.remainder(coords[:, 1], 2 * torch.pi)
#            _lo = torch.rad2deg(coords[:, 1].float())
#            print(f"C coords:     {tuple(coords.shape)} lon=[{_lo.min():.1f}, {_lo.max():.1f}]", flush=True)
            # Per-cell coords for QueryAggregationEngine (no query expansion)
            self.rope_cell_coords.data.copy_(coords)
#            _lo2 = torch.rad2deg(self.rope_cell_coords[:, 1].float())
#            print(f"D buffer:     lon=[{_lo2.min():.1f}, {_lo2.max():.1f}]", flush=True)
            coords = coords.unsqueeze(1).repeat(1, cf.ae_local_num_queries, 1)
            coords_flat = coords.flatten(0, 1).unsqueeze(0)
            offset = self.num_extra_tokens * cf.ae_local_num_queries
            self.rope_coords.data.fill_(0.0)
            self.rope_coords.data[:, offset : offset + coords_flat.shape[1], :].copy_(coords_flat)
#            _c = self.rope_cell_coords.detach().cpu()
#            print(f"ROPE cell: n={_c.shape[0]} lat=[{torch.rad2deg(_c[:,0]).min():.1f}, "
#                  f"{torch.rad2deg(_c[:,0]).max():.1f}] lon=[{torch.rad2deg(_c[:,1]).min():.1f}, "
#                  f"{torch.rad2deg(_c[:,1]).max():.1f}]", flush=True)

        # pe_global: always initialized. RoPE handles relative position in Q/K, but pe_global
        # provides per-cell token identity which is critical for masked cells that have no
        # content from local assimilation. Without it, masked cells are identical and the
        # teacher representation (evaluated without dropout) collapses to low rank.
#        self.pe_global.data.fill_(0.0)
#        xs = 2.0 * np.pi * torch.arange(0, dim_embed, 2, device=self.pe_global.device) / dim_embed
#        self.pe_global.data[..., 0::2] = 0.5 * torch.sin(
#            torch.outer(8 * torch.arange(cf.ae_local_num_queries, device=self.pe_global.device), xs)
#        )
#        self.pe_global.data[..., 0::2] += (
#            torch.sin(
#                torch.outer(torch.arange(self.num_healpix_cells, device=self.pe_global.device), xs)
#            )
#            .unsqueeze(1)
#            .repeat((1, cf.ae_local_num_queries, 1))
#        )
#        self.pe_global.data[..., 1::2] = 0.5 * torch.cos(
#            torch.outer(8 * torch.arange(cf.ae_local_num_queries, device=self.pe_global.device), xs)
#        )
#        self.pe_global.data[..., 1::2] += (
#            torch.cos(
#                torch.outer(torch.arange(self.num_healpix_cells, device=self.pe_global.device), xs)
#            )
#            .unsqueeze(1)
#            .repeat((1, cf.ae_local_num_queries, 1))
#        )
        # pe_global: always initialized. RoPE handles relative position in Q/K, but pe_global
        # provides per-cell token identity which is critical for masked cells that have no
        # content from local assimilation. Without it, masked cells are identical and the
        # teacher representation (evaluated without dropout) collapses to low rank.
        #
        # The encoding is GEOGRAPHIC, not index-based. The previous version used
        # sin(c * xs_k) on the compact cell index c; because active_cells is sorted, c is
        # monotone in the global nested index and hence in the level-5 parent, so the
        # lowest-frequency components (period ~1024 in c) were near-constant within each
        # coarse parent and showed up as visible blocks in the latent. A multi-scale
        # sin/cos encoding of (lat, lon) is continuous across the sphere, has no
        # hierarchical seams, and still gives every cell a distinct code.
        self.pe_global.data.fill_(0.0)
        xs = 2.0 * np.pi * torch.arange(0, dim_embed, 2, device=self.pe_global.device) / dim_embed

        # --- query-identity term (unchanged) -----------------------------------
        # With ae_local_num_queries == 1 this is a constant; kept so that >1 still works.
        self.pe_global.data[..., 0::2] = 0.5 * torch.sin(
            torch.outer(8 * torch.arange(cf.ae_local_num_queries, device=self.pe_global.device), xs)
        )
        self.pe_global.data[..., 1::2] = 0.5 * torch.cos(
            torch.outer(8 * torch.arange(cf.ae_local_num_queries, device=self.pe_global.device), xs)
        )

        # --- geographic cell-identity term -------------------------------------
        # Cell centres in the codebase's convention. r3tos2 returns azimuth wrapped to
        # (-pi, pi] by atan2; remainder() unwraps it to [0, 2pi) so that a domain
        # straddling the wrap is continuous. Recomputed here rather than reused from the
        # rope_2D block above, because pe_global is initialised unconditionally while
        # that block is not.
        pe_verts, _ = healpix_verts_rots(self.healpix_level, 0.5, 0.5)
        pe_verts = pe_verts[torch.from_numpy(self.domain.active_cells)]
        pe_coords = r3tos2(pe_verts.to(self.pe_global.device)).to(torch.float32)
        pe_lat = pe_coords[:, 0]
        pe_lon = torch.remainder(pe_coords[:, 1], 2 * torch.pi)

        # GLOBAL normalisation to [0, 1]: portable across domains, so a checkpoint's PE
        # means the same thing for a regional and a global run. lat in [-pi/2, pi/2],
        # lon in [0, 2pi).
        u_lat = (pe_lat + torch.pi / 2) / torch.pi
        u_lon = pe_lon / (2 * torch.pi)

        # Geometric frequency ladder from 1 cycle per globe up to ~4 cycles per healpix
        # cell, so the finest scale resolves individual cells while the coarsest is
        # smooth over the sphere. dim_embed is split 4 ways: sin/cos x lat/lon.
        n_freq = dim_embed // 4
        cell_frac = (58.6 / (2**self.healpix_level)) / 360.0
        max_freq = 4.0 / cell_frac
        freqs = torch.exp(
            torch.linspace(
                0.0, float(np.log(max_freq)), n_freq, device=self.pe_global.device
            )
        )

        ang_lat = 2 * torch.pi * torch.outer(u_lat, freqs)
        ang_lon = 2 * torch.pi * torch.outer(u_lon, freqs)
        pe_geo = torch.cat(
            [torch.sin(ang_lat), torch.cos(ang_lat), torch.sin(ang_lon), torch.cos(ang_lon)],
            dim=-1,
        )  # (num_healpix_cells, 4 * n_freq)

        assert pe_geo.shape[-1] == dim_embed, (
            f"geographic pe_global width {pe_geo.shape[-1]} != dim_embed {dim_embed}; "
            "ae_global_dim_embed must be divisible by 4."
        )

        self.pe_global.data += (
            pe_geo.to(self.pe_global.dtype).unsqueeze(1).repeat((1, cf.ae_local_num_queries, 1))
        )

        # healpix neighborhood structure (compact indexing, see __init__)
        temp = self.domain.neighbours_compact()
        # nbors *and* self
        self.hp_nbours.data[:, 0] = torch.arange(temp.shape[0], device=self.hp_nbours.device)
        self.hp_nbours.data[:, 1:] = torch.from_numpy(temp).to(self.hp_nbours.device)

        # precompute for varlen attention
        self.q_cells_lens.data.fill_(1)
        self.q_cells_lens.data[0] = 0

        # ensure all params have grad set to False

        return


class Model(torch.nn.Module):
    """WeatherGenerator model architecture

    WeatherGenerator consists of the following components:

    embeds: embedding networks: Stream specific embedding networks.

    ae_local_blocks: Local assimilation engine: transformer based network to combine different input
        streams per healpix cell.

    ae_adapter: Assimilation engine adapter: Adapter to transform local assimilation engine
        information to the global assimilation engine.

    ae_aggregation_blocks: Query aggregation engine: after the learnable queries are created per
        non-masked healpix cell, this engine combines information from all non-masked cells by
        using dense attention layers.

    ae_global_blocks: Global assimilation engine: Transformer network alternating between local and
        global attention based upon global attention density rate.

    fe_blocks: Forecasting engine: Transformer network using the output of global attention to
        advance the latent representation in time.

    embed_target_coords: Embedding networks for coordinates: Initializes embedding networks tailored
        for metadata embedded target coordinates. The architecture is either a linear layer or a
        multi-layer perceptron, determined by the configuration of the embedding target coordinate
        networks.

    pred_adapter_kv: Prediction adapter: Adapter to transform the global assimilation/forecasting
        engine output to the prediction engine. Uses an MLP if `cf.pred_adapter_kv` is True,
        otherwise it uses an identity function.

    target_token_engines: Prediction engine: Transformer based prediction network that generates
        output corresponding to target coordinates.

    pred_heads: Prediction head: Final layers using target token engines output for mapping target
        coordinates to its physical space.
    """

    def __init__(self, cf: Config, sources_size, targets_num_channels, targets_coords_size):
        """
        Args:
            cf : Configuration with model parameters
            sources_size : List of number of channels for models
            targets_num_channels : List with size of each output sample for coordinates target
                embedding
            targets_coords_size : List with size of each input sample for coordinates target
                embedding
        """
        super(Model, self).__init__()

        self.healpix_level = cf.healpix_level
        # Phase 2: single-level domain via the pyramid helper (see ModelParams).
        self.domain_pyramid = build_domain_pyramid(cf)
        self.num_healpix_cells = len(self.domain_pyramid.domain(self.domain_pyramid.finest))

        self.cf = cf
        self.dtype = get_dtype(self.cf.attention_dtype)
        self.sources_size = sources_size
        self.targets_num_channels = targets_num_channels
        self.targets_coords_size = targets_coords_size

        self.embed_target_coords = None
        self.encoder: EncoderModule | None = None
        self.forecast_engine: ForecastingEngine | IdentityEngine | None = None
        self.pred_heads = None
        self.q_cells: torch.Tensor | None = None
        self.streams: dict[str, typing.Any] = cf.streams
        self.target_token_engines = None

        assert cf.get("forecast", {}).get("att_dense_rate", 1.0) == 1.0, (
            "Local attention not adapted for register tokens"
        )
        self.num_register_tokens = cf.num_register_tokens
        self.latent_heads = None
        self.latent_pre_norm = None
        # auxiliary tokens
        self.class_token_idxs = list(
            range(cf.num_register_tokens, cf.num_register_tokens + cf.num_class_tokens)
        )
        self.register_token_idxs = list(range(cf.num_register_tokens))
        self.aux_token_idxs = list(range(cf.num_register_tokens + cf.num_class_tokens))
        self.num_aux_tokens = cf.num_register_tokens + cf.num_class_tokens

    def _create_latent_pred_head(
        self, global_cfg, name, loss_cfg, use_class_token, use_patch_token
    ):
        if loss_cfg["head"].lower() == "mlp":
            return LatentPredictionHeadMLP(
                name,
                global_cfg.ae_global_dim_embed,
                loss_cfg,
                use_class_token=use_class_token,
                use_patch_token=use_patch_token,
            )
        elif loss_cfg["head"].lower() == "transformer":
            return LatentPredictionHeadTransformer(
                global_cfg,
                name,
                global_cfg.ae_global_dim_embed,
                loss_cfg,
                use_class_token=use_class_token,
                use_patch_token=use_patch_token,
            )
        elif loss_cfg["head"].lower() == "identity":
            return LatentPredictionHeadIdentity()
        else:
            assert False, f"Unknown latent prediction head type {loss_cfg['head']}"

    def create(self) -> "Model":
        """Create each individual module of the model"""
        cf = self.cf

        self.encoder = EncoderModule(
            cf, self.sources_size, self.targets_num_channels, self.targets_coords_size
        )

        mode_cfg = cf.training_config
        if cf.fe_num_blocks > 0:
            self.forecast_engine = ForecastingEngine(cf, mode_cfg, self.num_healpix_cells)
        else:
            self.forecast_engine = IdentityEngine()

        # embed coordinates yielding one query token for each target token
        dropout_rate = cf.embed_dropout_rate
        self.embed_target_coords = torch.nn.ModuleDict()
        self.target_token_engines = torch.nn.ModuleDict()
        self.pred_heads = torch.nn.ModuleDict()

        # determine stream names once so downstream components use consistent keys
        loss_terms = [
            v.type for _, v in cf.training_config.losses.items() if v.get("enabled", True)
        ]
        if cf.validation_config.get("losses"):
            loss_terms += [
                v.type for _, v in cf.validation_config.losses.items() if v.get("enabled", True)
            ]

        if "LossPhysical" in loss_terms:
            for i_stream, (stream_name, si) in enumerate(self.streams.items()):
                # skip decoder if channels are empty
                if is_stream_forcing(si):
                    continue

                # skip for the moment to ensure target embedding and tte exist (ordering of
                # cf.streams is random)
                if si.get("pred_spatial_shared") is None:
                    # extract and setup relevant parameters
                    etc = si["embed_target_coords"]
                    tr = si["target_readout"]
                    num_layers = tr["num_layers"]
                    tr_mlp_hidden_factor = (
                        tr["mlp_hidden_factor"] if "mlp_hidden_factor" in tr else 2
                    )
                    tr_dim_head_proj = tr["dim_head_proj"] if "dim_head_proj" in tr else None
                    softcap = tr["softcap"] if "softcap" in tr else 0.0

                    dims_embed = [
                        si["embed_target_coords"]["dim_embed"] for _ in range(num_layers + 1)
                    ]

                    if is_root():
                        logger.info("{} :: coord embed: :: {}".format(si["name"], dims_embed))

                    dim_coord_in = self.targets_coords_size[i_stream]

                    # embedding network for coordinates
                    if etc["net"] == "linear":
                        self.embed_target_coords[stream_name] = NamedLinear(
                            f"embed_target_coords_{stream_name}",
                            in_features=dim_coord_in,
                            out_features=dims_embed[0],
                            bias=False,
                        )
                    elif etc["net"] == "mlp":
                        self.embed_target_coords[stream_name] = MLP(
                            dim_coord_in,
                            dims_embed[0],
                            hidden_factor=8,
                            with_residual=False,
                            dropout_rate=dropout_rate,
                            norm_eps=self.cf.mlp_norm_eps,
                            name=f"embed_target_coords_{stream_name}",
                        )
                    else:
                        assert False

                    if cf.decoder_type == "Linear":
                        tte = BilinearDecoder(
                            stream_name,
                            dims_embed[0],
                            cf.ae_global_dim_embed,
                            self.targets_num_channels[i_stream],
                        )
                    else:
                        # target prediction engines
                        # By default, decoder_type == "PerceiverIOCoordConditioning"
                        # routes to TargetPredictionEngineClassic; any other
                        # decoder_type routes to TargetPredictionEngine (which
                        # dispatches internally on decoder_type).
                        # Setting use_target_prediction_engine: true in the config
                        # forces TargetPredictionEngine also for
                        # PerceiverIOCoordConditioning (OriginalPredictionBlock path).
                        use_tpe = cf.get("use_target_prediction_engine", False)
                        tte_version = (
                            TargetPredictionEngine
                            if (cf.decoder_type != "PerceiverIOCoordConditioning" or use_tpe)
                            else TargetPredictionEngineClassic
                        )
                        if is_root():
                            logger.info(
                                f"{stream_name}: target readout engine = "
                                f"{tte_version.__name__} (decoder_type={cf.decoder_type})"
                            )
                        tte = tte_version(
                            cf,
                            dims_embed,
                            dim_coord_in,
                            tr_dim_head_proj,
                            tr_mlp_hidden_factor,
                            softcap,
                            stream_config=si,
                        )

                    self.target_token_engines[stream_name] = tte

                    # ensemble prediction heads to provide probabilistic prediction
                    final_activation = si["pred_head"].get("final_activation", "Identity")
                    if is_root():
                        logger.debug(
                            f"{final_activation} activation of pred head of {si['name']} stream"
                        )
                    self.pred_heads[stream_name] = EnsPredictionHead(
                        dims_embed[-1],
                        self.targets_num_channels[i_stream],
                        si["pred_head"]["num_layers"],
                        si["pred_head"]["ens_size"],
                        norm_type=cf.norm_type,
                        final_activation=final_activation,
                        stream_name=stream_name,
                    )

            # iterate again to setup shared spatial pred heads if specified in config
            for i_stream, (stream_name, si) in enumerate(self.streams.items()):
                # skip decoder if channels are empty
                if is_stream_forcing(si):
                    continue

                pred_spatial_shared = si.get("pred_spatial_shared")
                if pred_spatial_shared is not None:
                    if pred_spatial_shared not in self.streams.keys():
                        msg = f"Stream {stream_name} has pred_spatial_shared={pred_spatial_shared}"
                        msg += " but no stream with that name found."
                        raise ValueError(msg)
                    if pred_spatial_shared == stream_name:
                        msg = f"Stream {stream_name} has pred_spatial_shared={pred_spatial_shared}"
                        msg += "but cannot share with itself."
                        raise ValueError(msg)
                    logger.debug(
                        f"{stream_name} shares spatial prediction head with {pred_spatial_shared}."
                    )

                    self.embed_target_coords[stream_name] = self.embed_target_coords[
                        pred_spatial_shared
                    ]
                    self.target_token_engines[stream_name] = self.target_token_engines[
                        pred_spatial_shared
                    ]

                    assert pred_spatial_shared in self.streams.keys()
                    si_other = self.streams[pred_spatial_shared]
                    dims_embed = [
                        si_other["embed_target_coords"]["dim_embed"] for _ in range(num_layers + 1)
                    ]

                    # ensemble prediction heads to provide probabilistic prediction
                    final_activation = si["pred_head"].get("final_activation", "Identity")
                    if is_root():
                        logger.debug(
                            f"{final_activation} activation of pred head of {si['name']} stream"
                        )
                    self.pred_heads[stream_name] = EnsPredictionHead(
                        dims_embed[-1],
                        self.targets_num_channels[i_stream],
                        si["pred_head"]["num_layers"],
                        si["pred_head"]["ens_size"],
                        norm_type=cf.norm_type,
                        final_activation=final_activation,
                        stream_name=stream_name,
                    )

        # Latent heads for losses
        self.latent_heads = nn.ModuleDict()
        self.latent_pre_norm = nn.LayerNorm(cf.ae_global_dim_embed)

        ssl_losses_cfgs = [
            v
            for _, v in cf.training_config.losses.items()
            if v.type == "LossLatentSSLStudentTeacher" and v.get("enabled", True)
        ]

        # TODO: support multiple LossLatentSSLStudentTeacher terms
        assert len(ssl_losses_cfgs) <= 1, "To be implemented."
        for ssl_target_losses in ssl_losses_cfgs:
            self.latent_pre_norm = nn.LayerNorm(cf.ae_global_dim_embed)
            for loss, loss_conf in ssl_target_losses.loss_fcts.items():
                if loss == "iBOT":
                    self.latent_heads[loss] = self._create_latent_pred_head(
                        cf,
                        f"{loss}-head",
                        loss_conf,
                        use_class_token=True,
                        use_patch_token=True,
                    )
                elif loss == "JEPA":
                    self.latent_heads[loss] = self._create_latent_pred_head(
                        cf,
                        f"{loss}-head",
                        loss_conf,
                        use_class_token=False,
                        use_patch_token=True,
                    )
                elif loss == "DINO":
                    self.latent_heads[loss] = self._create_latent_pred_head(
                        cf,
                        f"{loss}-head",
                        loss_conf,
                        use_class_token=True,
                        use_patch_token=False,
                    )

        return self

    def reset_parameters(self):
        def _reset_params(module):
            if isinstance(module, nn.Linear | nn.LayerNorm):
                module.reset_parameters()
            else:
                pass

        self.apply(_reset_params)

    def print_num_parameters(self) -> None:
        """Print number of parameters for entire model and each module used to build the model"""

        num_params_embed = [
            get_num_parameters(self.encoder.embed_engine.embeds[name])
            for name in self.streams.keys()
        ]
        num_params_total = get_num_parameters(self)
        num_params_ae_local = get_num_parameters(self.encoder.ae_local_engine.ae_local_blocks)
        num_params_ae_global = get_num_parameters(self.encoder.ae_global_engine.ae_global_blocks)

        num_params_q_cells = (
            np.prod(self.encoder.q_cells.shape) if self.encoder.q_cells.requires_grad else 0
        )
        num_params_ae_adapter = get_num_parameters(self.encoder.ae_local_global_engine)

        num_params_ae_aggregation = get_num_parameters(
            self.encoder.ae_aggregation_engine.ae_aggregation_blocks
        )

        num_params_latent_heads = get_num_parameters(self.latent_heads)
        num_params_latent_heads += get_num_parameters(self.latent_pre_norm)

        num_params_fe = get_num_parameters(self.forecast_engine.fe_blocks)

        mdict = self.embed_target_coords
        num_params_embed_tcs = [
            get_num_parameters(mdict[name]) if mdict and name in mdict else 0
            for name in self.streams.keys()
        ]
        mdict = self.target_token_engines
        num_params_tte = [
            get_num_parameters(mdict[name]) if mdict and name in mdict else 0
            for name in self.streams.keys()
        ]
        mdict = self.pred_heads
        num_params_preds = [
            get_num_parameters(mdict[name]) if mdict and name in mdict else 0
            for name in self.streams.keys()
        ]

        print("-----------------")
        print(f"Total number of trainable parameters: {num_params_total:,}")
        print("Number of parameters:")
        print("  Embedding networks:")
        [
            print("    {} : {:,}".format(si["name"], np))
            for si, np in zip(self.streams.values(), num_params_embed, strict=False)
        ]
        print(f" Local assimilation engine: {num_params_ae_local:,}")
        print(f" Local-global adapter: {num_params_ae_adapter:,}")
        print(f" Learnable queries: {num_params_q_cells:,}")
        print(f" Query Aggregation engine: {num_params_ae_aggregation:,}")
        print(f" Global assimilation engine: {num_params_ae_global:,}")
        print(f" Latent prediction heads and pre-norm: {num_params_latent_heads:,}")
        print(f" Forecast engine: {num_params_fe:,}")
        print(" coordinate embedding, prediction networks and prediction heads:")
        zps = zip(
            self.streams.keys(),
            num_params_embed_tcs,
            num_params_tte,
            num_params_preds,
            strict=False,
        )
        for stream_name, np0, np1, np2 in zps:
            print(f"   {stream_name} : {np0:,} / {np1:,} / {np2:,}")
        print("-----------------")

    def tokens_to_latent_state(self, tokens_post_norm, tokens) -> LatentState:
        """
        Extract separate parts from global latent space representation and store in LatentState
        """
        toks_pn = tokens_post_norm
        return LatentState(
            register_tokens=toks_pn[:, self.register_token_idxs] if toks_pn is not None else None,
            class_token=toks_pn[:, self.class_token_idxs] if tokens_post_norm is not None else None,
            patch_tokens=toks_pn[:, self.num_aux_tokens :] if toks_pn is not None else None,
            z_pre_norm=tokens,
        )

    # ---------------------------------------------------------------- POU blend
    @staticmethod
    @functools.lru_cache(maxsize=4)
    def _pou_tables(healpix_level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute (once per healpix level, on CPU) the candidate table for the
        partition-of-unity readout blend.

        Returns
        -------
        cand : torch.LongTensor (num_cells, K)
            Column 0..8: the 1-ring of each cell (self first, matching hp_nbours).
            Columns 9..: extra pixels that the HEALPix bilinear interpolation
            scheme can select for points inside the cell but that lie outside
            the 1-ring (occurs only near polar / base-cell seams; at most 2 per
            cell, determined by dense subsampling).  Unused columns are padded
            with the cell itself (they receive zero weight and are skipped).
        pools : torch.LongTensor (K, num_cells, 9)
            pools[k, c] = the 1-ring of cand[c, k], i.e. the KV cell pool used
            when the readout is evaluated "as seen from" candidate k of cell c.
        """
        import astropy.units as _u
        from astropy.coordinates import Latitude as _Lat
        from astropy.coordinates import Longitude as _Lon
        from astropy_healpix import bilinear_interpolation_weights as _bil

        hl = healpix_level
        n = 12 * 4**hl

        with warnings.catch_warnings(action="ignore"):
            nb = hp.neighbours(np.arange(n), 2**hl, order="nested").transpose()
        for i, row in enumerate(nb):
            nb[i][row == -1] = i
        ring1 = np.hstack([np.arange(n).reshape(-1, 1), nb]).astype(np.int64)  # (n, 9)

        # dense subsampling of every cell (children at level hl+3, 64 per cell)
        # to discover which bilinear pixels can occur inside each cell
        sub = 3
        child = np.arange(n * 4**sub)
        lons, lats = hp.healpix_to_lonlat(child, 2 ** (hl + sub), dx=0.5, dy=0.5, order="nested")
        idx4, _ = _bil(
            _Lon(lons.rad, unit=_u.rad), _Lat(lats.rad, unit=_u.rad), nside=2**hl, order="nested"
        )
        idx4 = idx4.T.astype(np.int64)  # (num_children, 4)
        containing = astropy_healpix.healpy.ang2pix(
            2**hl, np.pi / 2 - lats.rad, lons.rad, nest=True
        ).astype(np.int64)

        max_extra = 2
        cand = np.concatenate(
            [ring1, np.repeat(np.arange(n).reshape(-1, 1), max_extra, axis=1)], axis=1
        )  # (n, 9 + max_extra), padded with self
        n_extra = np.zeros(n, dtype=np.int64)
        in_ring = (idx4[:, :, None] == ring1[containing][:, None, :]).any(-1)  # (children, 4)
        for c, pix in zip(containing[~in_ring.all(-1)], idx4[~in_ring.all(-1)]):
            for p in pix:
                if p not in cand[c, : 9 + n_extra[c]] and n_extra[c] < max_extra:
                    cand[c, 9 + n_extra[c]] = p
                    n_extra[c] += 1

        cand_t = torch.from_numpy(cand)                       # (n, K)
        pools = torch.from_numpy(ring1)[cand_t.T]             # (K, n, 9)
        return cand_t, pools

    def _pou_predict(
        self,
        stream_name: str,
        tc_tokens: torch.Tensor,
        t_coords: torch.Tensor,
        tcs_lens: torch.Tensor,
        tokens: torch.Tensor,
        s: list[int],
        batch_size: int,
    ) -> torch.Tensor:
        """
        Partition-of-unity readout blend removing healpix grid-imprint artifacts.

        Mechanism being fixed (hypothesis B, confirmed by the ablation run):
        the standard readout cross-attends only to the 9-cell pool of the
        containing cell c(p); the pool switches discretely at cell boundaries,
        which imprints the grid on the output.

        Construction
        ------------
        The HEALPix bilinear interpolation scheme assigns to every point p four
        pixels b_1..b_4(p) with weights lambda_1..lambda_4(p) >= 0 summing to 1
        that are CONTINUOUS functions of position over the whole sphere (this is
        the standard scheme used for HEALPix map interpolation).  We reuse these
        weights as the partition of unity:

            f(p) = sum_k  lambda_k(p) * f_{b_k(p)}(p)

        where f_j(p) is the existing readout evaluated with the KV pool centered
        on cell j (i.e. the 1-ring of j) instead of c(p).  Because the (pixel,
        weight) pairs vary continuously with p and do not reference c(p), the
        pool contribution to f(p) is continuous across every cell boundary by
        construction — the exiting pool's weight reaches zero exactly at the
        switch.  There is no tunable parameter.

        Implementation notes
        --------------------
        * The per-cell varlen attention structure of the standard readout is
          preserved exactly: for each candidate column k of the table from
          _pou_tables, one engine call is made whose KV gather mirrors the
          original tokens_nbors construction, with identical lens tensors.
          Columns whose weights are all zero in this batch are skipped
          (the 2 seam-extra columns are inactive unless the batch contains
          points in polar-seam cells).
        * The bilinear weights are computed on CPU via astropy_healpix from the
          smooth unit-sphere position stored in the last 3 channels of
          t_coords (requires the smooth-tail tokenizer patch; validated below).
        * Residual caveat: the coordinate features conditioning the decoder
          (the 99 cell-relative channels of t_coords) still switch with c(p).
          The ablation experiment showed these do not cause the artifacts, but
          for a fully boundary-free pipeline combine pou_blend with
          WEATHERGEN_ABLATE_CELL_COORDS=1 or a smooth coordinate encoding.

        Config:  pou_blend: true   (default false; no other parameters)
        Cost:    <= 9 (+2 near seams) readout calls instead of 1.
        """
        import astropy.units as _u
        from astropy.coordinates import Latitude as _Lat
        from astropy.coordinates import Longitude as _Lon
        from astropy_healpix import bilinear_interpolation_weights as _bil

        n = self.num_healpix_cells
        num_cells_total = batch_size * n
        dev = tc_tokens.device

        # ---- smooth unit-sphere position of every target point -------------
        p_r3 = t_coords[..., -3:].to(torch.float32)
        norms = p_r3.norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-2):
            raise ValueError(
                "pou_blend requires the smooth global R3 position in the last 3 "
                "channels of the target coordinates (tokenizer_utils patch with "
                "NUM_GLOBAL_COORD_CHANNELS). Found non-unit vectors instead."
            )

        # ---- containing cell of every point (from the packed varlen lens) --
        cell_of_point = torch.repeat_interleave(
            torch.arange(num_cells_total, device=dev), tcs_lens[1:].to(dev)
        )
        hp_cell = (cell_of_point % n).cpu()

        # ---- bilinear pixels and weights (CPU, vectorized) ------------------
        p_np = p_r3.detach().cpu().numpy()
        lon = np.arctan2(p_np[:, 1], p_np[:, 0])
        lat = np.arcsin(np.clip(p_np[:, 2], -1.0, 1.0))
        idx4, w4 = _bil(
            _Lon(lon, unit=_u.rad), _Lat(lat, unit=_u.rad), nside=2**self.healpix_level,
            order="nested",
        )
        idx4 = torch.from_numpy(idx4.T.astype(np.int64))       # (N_pts, 4)
        w4 = torch.from_numpy(w4.T.astype(np.float32))         # (N_pts, 4)

        # ---- map bilinear pixels onto candidate-table columns ---------------
        cand, pools = self._pou_tables(self.healpix_level)     # (n,K), (K,n,9)
        K = cand.shape[1]
        cand_rows = cand[hp_cell]                              # (N_pts, K)
        match = idx4.unsqueeze(-1) == cand_rows.unsqueeze(1)   # (N_pts, 4, K)
        found = match.any(-1)                                  # (N_pts, 4)
        col = match.to(torch.int64).argmax(-1)                 # (N_pts, 4)

        lam = torch.zeros(len(hp_cell), K, dtype=torch.float32)
        lam.scatter_add_(1, col, torch.where(found, w4, torch.zeros_like(w4)))
        n_missed = int((~found & (w4 > 1e-9)).sum())
        if n_missed > 0:
            # sampling gap in the candidate table: renormalize the matched mass
            # (continuity is broken only for these points; expected count: 0)
            logger.warning(
                f"pou_blend: {n_missed} bilinear pixel(s) missing from candidate "
                "table; weights renormalized for the affected points."
            )
        lam = lam / lam.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        lam = lam.to(dev)

        # ---- one readout call per active candidate column --------------------
        tokens_flat = tokens.reshape(s).flatten(0, 1)          # (B*n, nq, D)
        lens = torch.full(
            (num_cells_total + 1,), fill_value=9, dtype=torch.int32, device=dev
        )
        lens[0] = 0

        pred_blend = None
        for k in range(K):
            lam_k = lam[:, k]
            if not (lam_k > 0).any():
                continue
            # KV pool gather for column k, mirroring the original tokens_nbors
            # construction (incl. its per-batch index handling).
            idxs_k = pools[k].to(dev).unsqueeze(0).repeat((batch_size, 1, 1)).flatten(0, 1)

            def _slot_forward(tokens_flat_, tc_tokens_, t_coords_, idxs_flat_):
                # The gather is INSIDE the checkpoint so the large KV tensor
                # (same size as tokens_nbors) is freed after this slot's forward
                # and recomputed at backward. Without this, all K gathers plus
                # per-slot activations stay resident simultaneously (~K x the
                # readout memory), which OOMs on GPUs running near capacity.
                toks_k_ = tokens_flat_[idxs_flat_].flatten(0, 1)
                out_k_ = self.target_token_engines[stream_name](
                    latent=toks_k_,
                    output=tc_tokens_,
                    latent_lens=lens,
                    output_lens=tcs_lens,
                    coordinates=t_coords_,
                )
                return self.pred_heads[stream_name](out_k_)

            # Memory/compat trade-off: wrapping the slot in torch.utils.checkpoint
            # frees the large gathered KV between slots (recomputed at backward),
            # but under FSDP2 the backward-time recomputation runs outside the
            # FSDP hooks that unshard the DTensor parameters, producing
            # "got mixed torch.Tensor and DTensor" (nested non-reentrant
            # checkpointing + FSDP2 is not supported). So: checkpoint only when
            # the model is not FSDP-sharded; under FSDP2 call directly and rely
            # on the engine's internal per-layer checkpoints.
            if self.cf.get("with_fsdp", False) and self.cf.get("with_ddp", False):
                pred_k = _slot_forward(tokens_flat, tc_tokens, t_coords, idxs_k.flatten())
            else:
                pred_k = checkpoint(
                    _slot_forward,
                    tokens_flat,
                    tc_tokens,
                    t_coords,
                    idxs_k.flatten(),
                    use_reentrant=False,
                )                                               # (ens, N_pts, C)

            w_k = lam_k.view(1, -1, 1).to(pred_k.dtype)
            pred_blend = pred_k * w_k if pred_blend is None else pred_blend + pred_k * w_k

        return pred_blend


    def forward(self, model_params: ModelParams, batch: ModelBatch) -> ModelOutput:
        """Forward pass of the model

        Tokens are processed through the model components, which were defined in the create method.
        Args:
            model_params : Query and embedding parameters
            batch
        Returns:
            A list containing all prediction results
        """

        output = ModelOutput(batch.get_output_len())

        tokens, posteriors = self.encoder(model_params, batch)
        output.add_latent_prediction(0, "posteriors", posteriors)

        # recover batch dimension and separate input_steps
        shape = (len(batch), batch.get_num_source_steps(), *tokens.shape[1:])
        # collapse along input step dimension
        tokens = tokens.reshape(shape).sum(axis=1)

        # Allow for pushforward trick
        p_fwd = self.cf.training_config.get("forecast", {}).get("pushforward", False)
        # roll-out in latent space, iterate and generate output over requested output steps
        for step in batch.get_output_idxs():
            without_grad = p_fwd and self.training and step != max(batch.get_output_idxs())
            if without_grad:
                # Pushforward mode: advance tokens without grad; no decoding with torch.no_grad():
                tokens = self.forecast_engine(tokens, step, model_params.rope_coords)
                continue

            tokens = self.forecast_engine(tokens, step, model_params.rope_coords)
            # decoder predictions
            output = self.predict_decoders(model_params, step, tokens, batch, output)
            # latent predictions (raw and with SSL heads)
            output = self.predict_latent(model_params, step, tokens, batch, output)

        return output

    def predict_latent(
        self,
        model_params: ModelParams,
        step: int,
        tokens: torch.Tensor,
        batch: ModelBatch,
        output: ModelOutput,
    ) -> ModelOutput:
        """
        Compute latent predictions
        """

        # safe latent prediction
        tokens_post_norm = self.latent_pre_norm(tokens) if step == 0 else None
        latent_state = self.tokens_to_latent_state(tokens_post_norm, tokens)
        output.add_latent_prediction(step, "latent_state", latent_state)

        # latent predictions for SSL training
        for name, head in self.latent_heads.items():
            output.add_latent_prediction(step, name, head(latent_state))

        return output

    def predict_decoders(
        self,
        model_params: ModelParams,
        step: int,
        tokens: torch.Tensor,
        batch: ModelBatch,
        output: ModelOutput,
    ) -> ModelOutput:
        """
        Compute decoder-based predictions

        Predict outputs at the specific target coordinates based on the input weather state and
        pre-training task and projects the latent space representation back to physical space.

        Args:
            model_params : Query and embedding parameters
            fstep : Number of forecast steps
            tokens : Tokens from global assimilation engine
            streams_data : Used to initialize target coordinates tokens and index information
                List of StreamData len(streams_data) == batch_size_per_gpu
            target_coords_idxs : Indices of target coordinates
        Returns:
            Prediction output tokens in physical representation for each target_coords.
        """
        # Empty dicts evaluate to False in python
        if not self.pred_heads:
            return output

        # remove register  and class tokens
        tokens = tokens[:, self.num_aux_tokens :]

        # get 1-ring neighborhood for prediction
        batch_size = len(batch)
        s = [batch_size, self.num_healpix_cells, self.cf.ae_local_num_queries, tokens.shape[-1]]
        idxs = model_params.hp_nbours.unsqueeze(0).repeat((batch_size, 1, 1)).flatten(0, 1)
        tokens_nbors = tokens.reshape(s).flatten(0, 1)[idxs.flatten()].flatten(0, 1)
        # TODO: precompute in model_params?
        tokens_nbors_lens = torch.full(
            (s[0] * s[1] + 1,), fill_value=9, dtype=torch.int32, device=tokens_nbors.device
        )
        tokens_nbors_lens[0] = 0

        # pair with tokens from assimilation engine to obtain target tokens
        for stream_name in self.streams.keys():
            # extract target coords for current stream and fstep and convert to one tensor
            t_coords = [
                batch.samples[i_b].streams_data[stream_name].target_coords[step]
                for i_b in range(batch_size)
            ]
            t_coords_lens = [len(t) for t in t_coords]
            t_coords = torch.cat(t_coords)

            if len(t_coords) == 0:
                continue

            # embed token coords
            tc_embed = self.embed_target_coords[stream_name]
            tc_tokens = checkpoint(tc_embed, t_coords, use_reentrant=False)

            # skip when coordinate embeddings yields nan (i.e. the coord embedding network diverged)
            if torch.isnan(tc_tokens).any():
                logger.warning(
                    (
                        f"Skipping prediction for {stream_name} because",
                        f" of {torch.isnan(tc_tokens).sum()} NaN in tc_tokens.",
                    )
                )
                pred = torch.tensor([], device=tc_tokens.device)

            # skip empty lengths
            elif tc_tokens.shape[0] == 0:
                pred = torch.tensor([], device=tc_tokens.device)

            else:
                # lens for varlen attention
                tcls = torch.cat(
                    [
                        sample.streams_data[stream_name].target_coords_lens[step]
                        for sample in batch.samples
                    ]
                )
                tcs_lens = torch.cat([torch.zeros(1, dtype=torch.int32, device=tcls.device), tcls])

                if self.cf.decoder_type == "Linear":
                    pred = self.target_token_engines[stream_name](
                        tc_tokens,
                        tokens.reshape(-1, s[-1]),  # collapse the batch and token dimensions
                        tcs_lens,
                    ).unsqueeze(0)  # add ensemble dim: shape is then [1, preds_per_coord, channels]

                elif self.cf.get("pou_blend", False):
                    # Partition-of-unity readout blend: evaluates the readout
                    # under the (up to 4) HEALPix bilinear-interpolation pixels
                    # of each target point and combines with the bilinear
                    # weights, which are continuous across cell boundaries by
                    # construction. Removes the grid-imprint artifacts caused
                    # by the hard 9-cell pool switch at cell edges.
                    # Config: pou_blend: true  (no other parameters, no
                    # retraining required -- architecture is unchanged).
                    pred = self._pou_predict(
                        stream_name=stream_name,
                        tc_tokens=tc_tokens,
                        t_coords=t_coords,
                        tcs_lens=tcs_lens,
                        tokens=tokens,
                        s=s,
                        batch_size=batch_size,
                    )

                else:
                    tc_tokens = self.target_token_engines[stream_name](
                        latent=tokens_nbors,
                        output=tc_tokens,
                        latent_lens=tokens_nbors_lens,
                        output_lens=tcs_lens,
                        coordinates=t_coords,
                    )

                    # final prediction head to map back to physical space
                    pred = self.pred_heads[stream_name](tc_tokens)

            # recover batch dimension (ragged, so as list)
            pred = torch.split(pred, t_coords_lens, dim=1)
            output.add_physical_prediction(step, stream_name, pred)

        return output
