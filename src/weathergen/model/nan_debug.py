# TEMP diagnostic -- remove after debugging the attention-cascade NaN.
#
# Purpose: the forward NaN hook told us a Linear's WEIGHT is already NaN
# (shape 2048x2048, ae_global_dim_embed) while its INPUT is clean. That means the
# NaN is written during BACKWARD/optimizer, not created in forward. A forward hook
# can't see that. This module catches the NaN at its true origin:
#
#   1. scan_params_for_nan(model, "init")  -> is anything NaN already at init?
#      (rules the initialization hypothesis in/out)
#   2. register_grad_nan_hooks(model)      -> names the FIRST parameter whose
#      GRADIENT goes NaN during backward (the corrupting gradient).
#   3. check_params_after_step(model, step)-> called right after optimizer.step();
#      names the first WEIGHT that became NaN, at the exact step it happened.
#   4. enable_anomaly()                     -> torch.autograd.set_detect_anomaly:
#      makes backward throw a stack trace AT the op that creates the NaN gradient.
#
# Usage (minimal, in trainer.py):
#   from weathergen.model.nan_debug import (
#       scan_params_for_nan, register_grad_nan_hooks, check_params_after_step,
#       enable_anomaly,
#   )
#   ... after self.model is built (e.g. after init_model_and_shard in run()):
#       enable_anomaly()                       # optional, slow but definitive
#       scan_params_for_nan(self.model, "init")
#       register_grad_nan_hooks(self.model)
#   ... right after `self.grad_scaler.step(self.optimizer)` in train():
#       check_params_after_step(self.model, self.cf.general.istep)
#
# All prints are flushed and rank-prefixed via the process; grep for "NANDBG".

import torch

_grad_hook_fired = {"done": False}


def _rank():
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def scan_params_for_nan(model, tag: str) -> bool:
    """Scan all parameters for NaN/Inf. Returns True if any found. Prints names."""
    r = _rank()
    found = False
    for n, p in model.named_parameters():
        if p is None:
            continue
        # p may be a DTensor under FSDP; .isnan works, but guard just in case.
        try:
            bad_nan = torch.isnan(p).any().item()
            bad_inf = torch.isinf(p).any().item()
        except Exception as e:
            print(f"NANDBG[{r}] could not check param {n}: {e}", flush=True)
            continue
        if bad_nan or bad_inf:
            found = True
            kind = "NaN" if bad_nan else "Inf"
            print(
                f"NANDBG[{r}] {tag}: {kind} in param {n} shape={tuple(p.shape)}",
                flush=True,
            )
    if not found:
        print(f"NANDBG[{r}] {tag}: all params finite (no NaN/Inf).", flush=True)
    return found


def register_grad_nan_hooks(model):
    """
    Attach a hook to every parameter that fires the first time its gradient
    contains NaN/Inf, printing the parameter name. This runs during backward, so
    it names the corrupting gradient BEFORE it is applied by the optimizer.
    """
    r = _rank()
    count = 0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        def _mk(name):
            def _hook(grad):
                if _grad_hook_fired["done"]:
                    return grad
                try:
                    if grad is not None and (
                        torch.isnan(grad).any() or torch.isinf(grad).any()
                    ):
                        _grad_hook_fired["done"] = True
                        gmax = "nan"
                        try:
                            gmax = f"{grad.abs().max().item():.3e}"
                        except Exception:
                            pass
                        print(
                            f"NANDBG[{r}] FIRST NaN/Inf GRADIENT in param: {name} "
                            f"shape={tuple(grad.shape)} absmax={gmax}",
                            flush=True,
                        )
                except Exception as e:
                    print(f"NANDBG[{r}] grad hook error on {name}: {e}", flush=True)
                return grad

            return _hook

        p.register_hook(_mk(n))
        count += 1
    print(f"NANDBG[{r}] registered grad-NaN hooks on {count} params.", flush=True)


def check_params_after_step(model, step) -> bool:
    """
    Call right AFTER optimizer.step(). Names the first parameter that has become
    NaN/Inf, i.e. the moment a weight is corrupted. Returns True if any found.
    """
    r = _rank()
    for n, p in model.named_parameters():
        if p is None:
            continue
        try:
            if torch.isnan(p).any() or torch.isinf(p).any():
                print(
                    f"NANDBG[{r}] step={step}: param CORRUPTED after optimizer.step: "
                    f"{n} shape={tuple(p.shape)}",
                    flush=True,
                )
                return True
        except Exception as e:
            print(f"NANDBG[{r}] post-step check error on {n}: {e}", flush=True)
    return False


def enable_anomaly():
    """
    Turn on autograd anomaly detection: backward will raise with a stack trace at
    the exact op that first produces a NaN/Inf gradient. Slow -- use for one run.
    """
    torch.autograd.set_detect_anomaly(True)
    print(f"NANDBG[{_rank()}] torch.autograd anomaly detection ENABLED.", flush=True)
