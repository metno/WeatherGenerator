# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Opt-in encoder diagnostics: tokenizer, local assimilation, global assimilation and
training health, a few log lines per report.

Config:
    encoder_diag: True           # default False (no cost, nothing printed)
    encoder_diag_every: 200      # training steps between reports
    encoder_diag_update: True    # also measure the size of each optimizer update
                                 # (keeps a temporary copy of the local parameter shards
                                 # during one step every encoder_diag_every steps)

Lines printed by rank 0 (training: one step, averaged over ranks; validation: averaged
over the whole validation):

  [diag tok]    tokens per non-empty cell (mean, min..max over cells) and the fraction
                of cells that have tokens, per stream at each level
  [diag local]  latent after local assimilation: rms, spatial contrast (std across
                cells / rms) and sample change (|x - x_prev| / |x|, same rank, previous
                diagnosed sample: random dates in training, consecutive windows in
                validation)
  [diag global] latent after the global engine: change made by the engine
                (|out - in| / |in|), change made by the latent cascade, rms, sample
                change, effective dimension (participation ratio of the covariance
                over cells, and number of directions holding 90% of the variance, out
                of dim) and fraction of dead channels (std across cells < 1% of the
                median channel std)
  [diag noise]  stochastic latent: change made by the noise block and difference between
                two members, both relative to the latent size (compare with the 6-h sample
                change of [diag global] in validation)
  [diag grad]   per module group: gradient norm |g|, weight norm |w| and relative
                update |dw|/|w| of the optimizer step (healthy ~1e-4..1e-2;
                much smaller = not learning, much larger = unstable)

FSDP / DDP: nothing is created in the constructor; every rank computes local numbers;
the only collectives are in report(), called at points that every rank reaches
(training: every encoder_diag_every steps, same istep on every rank; validation: after
the loop, next to the existing loss logging). Keys are fixed from the config, so every
rank reduces the same tensors in the same order. Missing values count as 0 entries.
"""

import math

import torch
import torch.distributed as dist

_DIAG = None  # one per process (model and EMA copy write to the same object)

GRAD_GROUPS = (
    ("embed", ("embed_engine",)),
    ("local", ("_ae_local", "ae_local", "_ae_aggregation", "ae_aggregation", "_q_cells",
               "interpolator_latents")),
    ("global", ("_ae_global", "ae_global")),
    ("cascade", ("latent_cascade", "forecast_cascade")),
    ("forecast", ("_forecast_per_level", "forecast_engine")),
    ("decoder", ("embed_target_coords", "target_token_engines", "pred_heads")),
    ("noise", ("latent_noise",)),
)


def get_encoder_diag():
    return _DIAG


def _group_of(name: str) -> str:
    for g, keys in GRAD_GROUPS:
        if any(k in name for k in keys):
            return g
    return "other"


def _local(t):
    # FSDP2 parameters and gradients are DTensors: use this rank's shard
    return t.to_local() if hasattr(t, "to_local") else t


class EncoderDiag:
    def __init__(self, cf, levels, stream_encode_level):
        self.enabled = bool(cf.get("encoder_diag", False))
        self.every = max(int(cf.get("encoder_diag_every", 200)), 1)
        self.with_update = bool(cf.get("encoder_diag_update", True))
        self.levels = [int(lv) for lv in levels]
        self.streams = {
            lv: [s for s, l in stream_encode_level.items() if int(l) == lv] for lv in self.levels
        }
        self.dim = int(cf.ae_global_dim_embed)
        self.active = False
        self.training = True
        self._prev = {}
        self._snap = None
        # fixed key list: identical on every rank
        keys = []
        for lv in self.levels:
            for s in self.streams[lv]:
                keys += [f"tok/{lv}/{s}/{k}" for k in ("mean", "min", "max", "cover")]
            keys += [f"local/{lv}/{k}" for k in ("rms", "xcell", "dsample")]
            keys += [f"global/{lv}/{k}" for k in
                     ("change", "cascade", "rms", "dsample", "effdim", "dim90", "dead")]
            keys += [f"noise/{lv}/{k}" for k in ("change", "spread")]
        for g, _ in GRAD_GROUPS + (("other", ()),):
            keys += [f"grad/{g}/{k}" for k in ("g2", "w2", "u2")]
        self.keys = keys
        self._reset()

    # ------------------------------------------------------------------ state
    def _reset(self):
        self.acc = {k: [0.0, 0.0, math.inf, -math.inf] for k in self.keys}

    def _add(self, key, v):
        if key not in self.acc or v is None:
            return
        v = float(v)
        if not math.isfinite(v):
            return
        a = self.acc[key]
        a[0] += v
        a[1] += 1.0
        a[2] = min(a[2], v)
        a[3] = max(a[3], v)

    def due(self, istep: int) -> bool:
        return self.enabled and istep % self.every == 0

    def begin_step(self, istep: int, training: bool):
        """Call before the forward pass. Training: diagnose only every `every` steps."""
        if not self.enabled:
            self.active = False
            return
        self.training = training
        if training:
            self.active = self.due(istep)
            if self.active:
                self._reset()
        else:
            self.active = True

    def begin_validation(self):
        if self.enabled:
            self._reset()
            self.training = False

    # ---------------------------------------------------------- forward hooks
    @torch.no_grad()
    def tokens(self, level, tokens_lens_l):
        """tokens_lens_l: (steps, samples, streams_at_level, cells)."""
        if not self.active or tokens_lens_l is None or tokens_lens_l.dim() != 4:
            return
        names = self.streams.get(int(level), [])
        n = min(len(names), tokens_lens_l.shape[2])
        for i in range(n):
            c = tokens_lens_l[:, :, i, :].reshape(-1).float()
            if c.numel() == 0:
                continue
            nz = c > 0
            pre = f"tok/{int(level)}/{names[i]}/"
            self._add(pre + "cover", nz.float().mean().item())
            if bool(nz.any()):
                cz = c[nz]
                self._add(pre + "mean", cz.mean().item())
                self._add(pre + "min", cz.min().item())
                self._add(pre + "max", cz.max().item())

    @torch.no_grad()
    def latent(self, kind, level, x, n_cell_tokens, ref=None):
        """
        kind: "local" (after local assimilation), "global" (after the global engine;
        ref = its input) or "cascade" (after the latent cascade; ref = its input).
        x: (batch, aux + cells * queries, dim); the last n_cell_tokens are the cells.
        Only sample 0 is analysed.
        """
        if not self.active or x is None or x.dim() != 3 or x.shape[0] == 0:
            return
        lv = int(level)
        with torch.autocast(device_type=x.device.type, enabled=False):
            n0 = x.shape[1] - int(n_cell_tokens)
            c = x[0, n0:].float()                                         # (N, D)
            if c.numel() == 0:
                return
            if ref is not None and ref.shape == x.shape:
                r = ref[0, n0:].float()
                ch = ((c - r).norm() / r.norm().clamp(min=1e-12)).item()
                self._add(f"global/{lv}/{'change' if kind == 'global' else 'cascade'}", ch)
            if kind == "cascade":
                return
            rms = c.pow(2).mean().sqrt()
            self._add(f"{kind}/{lv}/rms", rms.item())
            if kind == "local":
                self._add(f"local/{lv}/xcell",
                          (c.std(dim=0).pow(2).mean().sqrt() / rms.clamp(min=1e-12)).item())
            pkey = (kind, lv)
            prev = self._prev.get(pkey)
            if prev is not None and prev.shape == c.shape and prev.device == c.device:
                self._add(f"{kind}/{lv}/dsample",
                          ((c - prev).norm() / c.norm().clamp(min=1e-12)).item())
            self._prev[pkey] = c.detach().clone()
            if kind == "global":
                xc = c - c.mean(dim=0, keepdim=True)
                if xc.shape[0] > 8192:   # bound the cost of the SVD
                    xc = xc[:: math.ceil(xc.shape[0] / 8192)]
                lam = torch.linalg.svdvals(xc).pow(2)
                if lam.sum() > 0:
                    self._add(f"global/{lv}/effdim", (lam.sum() ** 2 / lam.pow(2).sum()).item())
                    frac = torch.cumsum(lam, 0) / lam.sum()
                    self._add(f"global/{lv}/dim90", int((frac < 0.9).sum().item()) + 1)
                sd = c.std(dim=0)
                med = sd.median().clamp(min=1e-12)
                self._add(f"global/{lv}/dead", (sd < 0.01 * med).float().mean().item())

    @torch.no_grad()
    def noise(self, level, x_in, x_out, member, n_cell_tokens):
        """
        Stochastic latent. member 0: change made by the noise block, |out - in| / |in|;
        member 1: difference between two members, |out_1 - out_0| / |in|. Both on the cells
        of sample 0; compare them with the 6-h "sample change" of [diag global] (validation).
        """
        if not self.active or x_in is None or x_in.dim() != 3:
            return
        lv = int(level)
        with torch.autocast(device_type=x_in.device.type, enabled=False):
            n0 = x_in.shape[1] - int(n_cell_tokens)
            a = x_in[0, n0:].float()
            b = x_out[0, n0:].float()
            ref = a.norm().clamp(min=1e-12)
            if member == 0:
                self._add(f"noise/{lv}/change", ((b - a).norm() / ref).item())
                self._noise_m0 = b.detach().clone()
            elif member == 1:
                m0 = getattr(self, "_noise_m0", None)
                if m0 is not None and m0.shape == b.shape:
                    self._add(f"noise/{lv}/spread", ((b - m0).norm() / ref).item())
                self._noise_m0 = None

    # ------------------------------------------------------- optimizer hooks
    @torch.no_grad()
    def pre_opt(self, model, istep):
        """Call after gradient clipping, before optimizer.step(). Every rank."""
        if not self.due(istep):
            return
        ws = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self._grad_acc = {}
        self._snap = [] if self.with_update else None
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            g = _group_of(name)
            # replicated (non-sharded) tensors are counted once over all ranks
            f = 1.0 if hasattr(p, "to_local") else 1.0 / ws
            pl = _local(p)
            a = self._grad_acc.setdefault(g, [0.0, 0.0, 0.0])
            a[1] += f * pl.float().pow(2).sum().item()
            if p.grad is not None:
                a[0] += f * _local(p.grad).float().pow(2).sum().item()
            if self._snap is not None:
                self._snap.append((g, f, p, pl.detach().clone()))

    @torch.no_grad()
    def post_opt(self, model, istep):
        """Call after optimizer.step(). Every rank."""
        if not self.due(istep):
            return
        acc = getattr(self, "_grad_acc", {})
        if self._snap is not None:
            for g, f, p, old in self._snap:
                acc[g][2] += f * (_local(p).float() - old.float()).pow(2).sum().item()
            self._snap = None
        for g, (g2, w2, u2) in acc.items():
            self._add(f"grad/{g}/g2", g2)
            self._add(f"grad/{g}/w2", w2)
            if self.with_update:
                self._add(f"grad/{g}/u2", u2)

    # ----------------------------------------------------------------- report
    def _reduce(self, device):
        n = len(self.keys)
        s = torch.tensor([self.acc[k][0] for k in self.keys] + [self.acc[k][1] for k in self.keys],
                         dtype=torch.float64, device=device)
        mn = torch.tensor([self.acc[k][2] for k in self.keys], dtype=torch.float64, device=device)
        mx = torch.tensor([self.acc[k][3] for k in self.keys], dtype=torch.float64, device=device)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(s, op=dist.ReduceOp.SUM)
            dist.all_reduce(mn, op=dist.ReduceOp.MIN)
            dist.all_reduce(mx, op=dist.ReduceOp.MAX)
        s, mn, mx = s.cpu().tolist(), mn.cpu().tolist(), mx.cpu().tolist()
        return {k: (s[i], s[n + i], mn[i], mx[i]) for i, k in enumerate(self.keys)}

    def report(self, tag, step, device=None):
        """
        Collective: call on EVERY rank at the same point. Returns the lines printed by
        rank 0 (empty list elsewhere). For training call it when due(istep) is True.
        """
        if not self.enabled:
            return []
        if device is None:
            nccl = (dist.is_available() and dist.is_initialized()
                    and dist.get_backend() == "nccl")
            device = torch.device("cuda", torch.cuda.current_device()) if nccl else "cpu"
        r = self._reduce(device)
        self._reset()
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return []

        def mean(k):
            s, c, _, _ = r[k]
            return s / c if c > 0 else float("nan")

        def tot(k):  # summed over ranks (grad sums of squares), one entry per rank
            s, c, _, _ = r[k]
            ws = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
            return s / max(c / ws, 1.0) if c > 0 else float("nan")

        lines = []
        hdr = f"[diag {{}}] {tag} step {step}"
        for lv in self.levels:
            parts = []
            for s in self.streams[lv]:
                k = f"tok/{lv}/{s}/"
                if r[k + "cover"][1] == 0:
                    continue
                if r[k + "mean"][1] == 0:
                    parts.append(f"{s} no tokens")
                    continue
                parts.append(f"{s} {mean(k + 'mean'):.1f} ({r[k + 'min'][2]:.0f}..{r[k + 'max'][3]:.0f})"
                             f" cover {100 * mean(k + 'cover'):.0f}%")
            if parts:
                lines.append(f"{hdr.format('tok')} hl{lv}: tokens/cell " + " | ".join(parts))
        for lv in self.levels:
            k = f"local/{lv}/"
            if r[k + "rms"][1] == 0:
                continue
            lines.append(f"{hdr.format('local')} hl{lv}: rms {mean(k + 'rms'):.3g} | "
                         f"across cells {mean(k + 'xcell'):.3g} | "
                         f"sample change {mean(k + 'dsample'):.3g}")
        for lv in self.levels:
            k = f"global/{lv}/"
            if r[k + "rms"][1] == 0 and r[k + "change"][1] == 0:
                continue
            lines.append(f"{hdr.format('global')} hl{lv}: engine change {mean(k + 'change'):.3g} | "
                         f"cascade change {mean(k + 'cascade'):.3g} | rms {mean(k + 'rms'):.3g} | "
                         f"sample change {mean(k + 'dsample'):.3g} | "
                         f"eff. dims {mean(k + 'effdim'):.0f} (90%: {mean(k + 'dim90'):.0f}) "
                         f"of {self.dim} | dead {100 * mean(k + 'dead'):.0f}%")
        for lv in self.levels:
            k = f"noise/{lv}/"
            if r[k + "change"][1] == 0:
                continue
            lines.append(f"{hdr.format('noise')} hl{lv}: change by noise block "
                         f"{mean(k + 'change'):.3g} | between members {mean(k + 'spread'):.3g}")
        if self.training:
            parts = []
            for g, _ in GRAD_GROUPS + (("other", ()),):
                k = f"grad/{g}/"
                if r[k + "w2"][1] == 0 or tot(k + "w2") <= 0:
                    continue
                w = math.sqrt(tot(k + "w2"))
                gn = math.sqrt(max(tot(k + "g2"), 0.0))
                p = f"{g} |g| {gn:.2e} |w| {w:.1f}"
                if self.with_update and r[k + "u2"][1] > 0:
                    p += f" upd {math.sqrt(max(tot(k + 'u2'), 0.0)) / w:.1e}"
                parts.append(p)
            if parts:
                lines.append(f"{hdr.format('grad')}: " + " | ".join(parts))
        for ln in lines:
            print(ln, flush=True)
        return lines


def setup_encoder_diag(cf, levels, stream_encode_level):
    """Create the per-process diagnostics object (no tensors). Returns it."""
    global _DIAG
    if _DIAG is None:
        _DIAG = EncoderDiag(cf, levels, stream_encode_level)
    return _DIAG
