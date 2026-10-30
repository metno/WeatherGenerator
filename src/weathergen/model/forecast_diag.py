# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Opt-in diagnostics for the latent forecast step (config: ``forecast_diag: True``).

Question answered: does the forecast engine shrink the latent towards its mean (variance
collapse) and/or smooth it in space, so that the decoder is fed latents that are blurrier
than the real latents it was trained on?

Everything is computed on the latent *before* (z_in, the encoded state at t0) and *after*
(z_out, the forecast for t0 + dt) the forecast engine, in evaluation mode only, from the
validation samples the run already uses. No target-time encoding is needed: over a validation
set that covers all times of day, the distribution of z_in is the distribution of the real
latent at t0 + dt (same states shifted by dt), so z_in is the reference for z_out.

Reported per latent level (accumulated over the validation epoch, reset when training resumes):
  rel_update  ||z_out - z_in|| / ||z_in||. ~0: engine ~ persistence. Large: big changes.
  cos         mean per-cell cosine(z_in, z_out). 1 = unchanged direction.
  var_ratio   temporal variance (across samples, per cell and channel, averaged) of z_out over
              that of z_in. A real forecast latent should be ~1; clearly < 1 means the engine
              regresses towards the mean.
  rough_ratio spatial roughness (distance of each cell from the mean of its 9-cell ring,
              normalised by the spatial variance) of z_out over that of z_in. < 1 means the
              engine spatially smooths the latent.
  blocks      mean relative change ||delta|| / ||x|| made by each block of the engine.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class ForecastLatentDiag:
    def __init__(self, log_every: int = 16, log: bool = True):
        self.log_every = log_every
        self.log = log
        self.acc = {}

    def reset(self) -> None:
        self.acc.clear()

    @staticmethod
    def _rough(x: torch.Tensor, nbours: torch.Tensor):
        """Sum of squared distances to the 9-cell ring mean, and total spatial energy.

        x: (B, N, D); nbours: (N, K) compact indices (self + neighbours).
        """
        ring = torch.zeros_like(x)
        for k in range(nbours.shape[1]):  # loop: avoids a (B, N, K, D) temporary
            ring += x[:, nbours[:, k]]
        ring /= nbours.shape[1]
        high = ((x - ring) ** 2).sum()
        energy = ((x - x.mean(dim=1, keepdim=True)) ** 2).sum()
        return high, energy

    @torch.no_grad()
    def update(self, key, z_in, z_out, num_aux, ncells, nq, nbours, block_rel=None):
        """z_in, z_out: (B, num_aux + ncells * nq, D) latents of one level."""
        B, D = z_in.shape[0], z_in.shape[-1]

        def cells(z):  # (B, ncells, D); queries averaged if nq > 1
            return z[:, num_aux:].float().reshape(B, ncells, nq, D).mean(2)

        x0, x1 = cells(z_in), cells(z_out)

        a = self.acc.get(key)
        if a is None:
            a = self.acc[key] = {
                "n": 0, "calls": 0,
                "d2": 0.0, "n02": 0.0,
                "cos": 0.0, "cos_n": 0,
                "s0": torch.zeros(ncells, D, device=x0.device, dtype=torch.float32),
                "q0": torch.zeros(ncells, D, device=x0.device, dtype=torch.float32),
                "s1": torch.zeros(ncells, D, device=x0.device, dtype=torch.float32),
                "q1": torch.zeros(ncells, D, device=x0.device, dtype=torch.float32),
                "h0": 0.0, "e0": 0.0, "h1": 0.0, "e1": 0.0,
                "blk": None, "blk_names": None,
            }

        a["n"] += B
        a["calls"] += 1
        a["d2"] += ((x1 - x0) ** 2).sum().item()
        a["n02"] += (x0**2).sum().item()
        a["cos"] += torch.nn.functional.cosine_similarity(x0, x1, dim=-1).sum().item()
        a["cos_n"] += B * ncells
        a["s0"] += x0.sum(0)
        a["q0"] += (x0**2).sum(0)
        a["s1"] += x1.sum(0)
        a["q1"] += (x1**2).sum(0)
        h0, e0 = self._rough(x0, nbours)
        h1, e1 = self._rough(x1, nbours)
        a["h0"] += h0.item()
        a["e0"] += e0.item()
        a["h1"] += h1.item()
        a["e1"] += e1.item()

        if block_rel:
            vals = [v for _, v in block_rel]
            if a["blk"] is None:
                a["blk"] = [0.0] * len(vals)
                a["blk_names"] = [n for n, _ in block_rel]
            for i, v in enumerate(vals):
                a["blk"][i] += v

        if self.log and a["calls"] % self.log_every == 0:
            # print, not logger: independent of the logging configuration of the run
            print(self.summary(key), flush=True)

    def stats(self, key) -> dict:
        a = self.acc[key]
        n = max(a["n"], 1)
        var0 = (a["q0"] / n - (a["s0"] / n) ** 2).clamp_min(0).mean().item()
        var1 = (a["q1"] / n - (a["s1"] / n) ** 2).clamp_min(0).mean().item()
        r0 = a["h0"] / max(a["e0"], 1e-30)
        r1 = a["h1"] / max(a["e1"], 1e-30)
        out = {
            "n": a["n"],
            "rel_update": (a["d2"] / max(a["n02"], 1e-30)) ** 0.5,
            "cos": a["cos"] / max(a["cos_n"], 1),
            "var_in": var0,
            "var_out": var1,
            "var_ratio": var1 / max(var0, 1e-30),
            "rough_in": r0,
            "rough_out": r1,
            "rough_ratio": r1 / max(r0, 1e-30),
        }
        if a["blk"] is not None:
            out["blocks"] = [
                (nm, v / a["calls"]) for nm, v in zip(a["blk_names"], a["blk"], strict=True)
            ]
        return out

    def summary(self, key) -> str:
        s = self.stats(key)
        txt = (
            f"[fcdiag] {key} n={s['n']} | rel_update={s['rel_update']:.4f} cos={s['cos']:.4f} | "
            f"var_ratio={s['var_ratio']:.3f} (in {s['var_in']:.3e} out {s['var_out']:.3e}) | "
            f"rough_ratio={s['rough_ratio']:.3f} (in {s['rough_in']:.3e} out {s['rough_out']:.3e})"
        )
        if "blocks" in s:
            txt += " | blocks: " + " ".join(f"{nm[:6]}={v:.4f}" for nm, v in s["blocks"])
        return txt
