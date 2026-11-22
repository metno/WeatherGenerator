# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Opt-in timing of one training step, split into its parts.

    timing_diag: True          # default False
    timing_diag_every: 50      # training steps between timed steps

On a timed step every part ends with a device synchronisation (so the step itself is a bit
slower); rank 0 prints one line with its own times, for example

    [timing] step 50 (rank 0): encoder 1.9 s | members 2 x 3.1 s | loss 0.9 s |
             backward 8.0 s | optimizer 0.3 s | total 17.3 s

No collectives: each rank measures itself; only rank 0 prints.
"""

import time

import torch
import torch.distributed as dist

_TIMER = None


def get_step_timer():
    return _TIMER


def setup_step_timer(cf):
    global _TIMER
    if _TIMER is None and bool(cf.get("timing_diag", False)):
        _TIMER = StepTimer(int(cf.get("timing_diag_every", 50)))
    return _TIMER


class StepTimer:
    def __init__(self, every: int):
        self.every = max(int(every), 1)
        self.active = False
        self.acc = {}
        self.count = {}
        self._t = 0.0
        self._t0 = 0.0
        self.istep = 0

    @staticmethod
    def _sync():
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    def begin(self, istep: int, training: bool = True):
        self.active = training and istep % self.every == 0
        if self.active:
            self.istep = istep
            self.acc, self.count = {}, {}
            self._sync()
            self._t = self._t0 = time.perf_counter()

    def mark(self, name: str):
        if not self.active:
            return
        self._sync()
        now = time.perf_counter()
        self.acc[name] = self.acc.get(name, 0.0) + (now - self._t)
        self.count[name] = self.count.get(name, 0) + 1
        self._t = now

    def end(self):
        if not self.active:
            return None
        self._sync()
        total = time.perf_counter() - self._t0
        self.active = False
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank != 0:
            return None
        parts = []
        for name, t in self.acc.items():
            n = self.count[name]
            parts.append(f"{name} {n} x {t / n:.2f} s" if n > 1 else f"{name} {t:.2f} s")
        line = f"[timing] step {self.istep} (rank 0): " + " | ".join(parts) + f" | total {total:.2f} s"
        print(line, flush=True)
        return line
