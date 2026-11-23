"""target_source_correspondence in a LossPhysical loss term (CPU):
  - the key is read by the Masker (source / target relationship) ...
  - ... and is NOT passed on to the loss functions (mse, Haar, energy would raise TypeError)."""
import torch
from omegaconf import OmegaConf

from weathergen.train.loss_modules import loss_functions
from weathergen.train.loss_modules.loss_module_physical import LossPhysical

corr = {0: {0: "independent"}}
loss_fcts = {
    "mse": {"weight": 0.001, "target_source_correspondence": corr},
    "global_haar_wavelet_reshape_varweighted": {
        "weight": 1, "detail_weight": 2.0, "num_levels": 4, "target_source_correspondence": corr,
    },
    "global_haar_energy_reshape": {
        "weight": 0.01, "num_levels": 4, "target_source_correspondence": corr,
    },
}
mode_cfg = OmegaConf.create({"forecast": {"offset": 0}})
lp = LossPhysical(OmegaConf.create({"streams": {}}), mode_cfg, "train", "cpu", **loss_fcts)

for fn, w, name, params in lp.loss_fcts:
    print(f"{name:45s} weight {w:<6} params {sorted(params)}")
    assert "target_source_correspondence" not in params
    assert "weight" not in params

# the mse term can be called with its params (it raised TypeError before the fix)
fn, _, _, params = lp.loss_fcts[0]
t, p = torch.randn(10, 3), torch.randn(1, 10, 3)
loss, _ = fn(t, p, None, None, **params)
print(f"mse call ok: {loss.item():.3f}")
try:
    loss_functions.mse(t, p, None, None, target_source_correspondence=corr)
    raise AssertionError("expected TypeError")
except TypeError:
    print("unfiltered key -> TypeError (as expected)")
print("OK")
