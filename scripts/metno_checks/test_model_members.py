"""Model.forward with the member loop (stand-in encoder / decoders, CPU):
   noise off  -> identical to the original forward of the branch;
   noise on, zero start -> every member identical to the deterministic prediction;
   noise on, trained -> members differ, stacked on dim 0."""
import subprocess
import sys
import types

import numpy as np
import torch

import weathergen.model.model as mm
from weathergen.model.stochastic import LatentNoise

# original Model.forward from the branch before the stochastic latent (commit e57b481);
# usage: python test_model_members.py <repo> [commit]
repo = sys.argv[1]
commit = sys.argv[2] if len(sys.argv) > 2 else "e57b481"
src = subprocess.run(["git", "-C", repo, "show", f"{commit}:src/weathergen/model/model.py"],
                     capture_output=True, text=True, check=True).stdout
a = src.index("    def forward(self, model_params: ModelParams, batch: ModelBatch) -> ModelOutput:")
b = src.index("    def predict_latent(", a)
import textwrap
ns = dict(vars(mm))
exec(textwrap.dedent(src[a:b]), ns)
old_forward = ns["forward"]


class Cf(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)

    def __getattr__(self, k):
        return self[k]


D, C = 32, 3
cells8 = np.arange(1024, 1024 + 64, dtype=np.int64)
ncell = {5: 4, 8: 64}
nb = torch.stack([torch.arange(64)] + [torch.roll(torch.arange(64), s) for s in (1, -1, 2, -2, 3, -3, 4, -4)], 1)
mp8 = types.SimpleNamespace(hp_nbours=nb.int(), domain=types.SimpleNamespace(active_cells=cells8),
                            rope_coords=None)
mp5 = types.SimpleNamespace(rope_coords=None)
model_params = types.SimpleNamespace(params_for=lambda lvl: mp8 if lvl == 8 else mp5, rope_coords=None)


def make_cf(stoch):
    return Cf(training_config=Cf(forecast={}), ae_global_dim_embed=D, ae_local_num_queries=1,
              num_register_tokens=0, num_class_tokens=0, norm_eps=1e-5,
              stochastic=dict(enabled=stoch, noise_levels=[6, 7], noise_dim=4, smooth_passes=1,
                              att_dim=16, att_heads=4, members_train=2, members_val=3))


class Pyr:
    finest = 8

    def domain(self, lvl):
        return list(range(ncell[lvl]))


W = torch.randn(D, C)


def make_self(cf, noise):
    s = types.SimpleNamespace()
    s.cf, s.training, s.domain_pyramid = cf, True, Pyr()
    g = torch.Generator().manual_seed(3)
    lat = {5: torch.randn(1, 4, D, generator=g), 8: torch.randn(1, 64, D, generator=g)}
    s.encoder = lambda mp, batch: (lat, {5: None, 8: None})
    s._forecast_per_level = None
    s.forecast_engine = lambda t, step, rope: t
    s.forecast_cascade = lambda d: d
    s.num_aux_tokens = 0

    def predict_decoders(model_params, step, tokens, batch, output, tokens_lvl=None):
        # "decoder": per-point prediction from hl8 + hl5 tokens, 2 samples of 40 / 24 points
        y = (tokens_lvl[8][0] @ W + tokens_lvl[5][0].mean(0) @ W).unsqueeze(0)   # (1, 64, C)
        output.add_physical_prediction(step, "CARRA2", torch.split(y, [40, 24], dim=1))
        return output

    s.predict_decoders = predict_decoders
    s.predict_latent = lambda mp, step, tokens, batch, output: (
        output.add_latent_prediction(step, "latent_state", tokens) or output)
    s.latent_noise = noise
    return s


batch = types.SimpleNamespace(get_output_len=lambda: 1, get_num_source_steps=lambda: 1,
                              get_output_idxs=lambda: [0], __len__=lambda self=None: 1)
batch = type("B", (), {"get_output_len": lambda self: 1, "get_num_source_steps": lambda self: 1,
                       "get_output_idxs": lambda self: [0], "__len__": lambda self: 1})()

# noise off: identical to the original forward
out_old = old_forward(make_self(make_cf(False), None), model_params, batch)
out_new = mm.Model.forward(make_self(make_cf(False), None), model_params, batch)
for i in range(2):
    assert torch.equal(out_old.physical[0]["CARRA2"][i], out_new.physical[0]["CARRA2"][i])
assert torch.equal(out_old.latent[0]["latent_state"], out_new.latent[0]["latent_state"])
print("noise off: identical to the original Model.forward")

# noise on, zero start: 2 members (training), each identical to the deterministic prediction
torch.manual_seed(0)
noise = LatentNoise(make_cf(True), 8)
noise.train()
out_z = mm.Model.forward(make_self(make_cf(True), noise), model_params, batch)
for i, n in enumerate((40, 24)):
    p = out_z.physical[0]["CARRA2"][i]
    assert p.shape == (2, n, C), p.shape
    for m in range(2):
        assert torch.equal(p[m], out_old.physical[0]["CARRA2"][i][0])
print("noise on, zero start: shape [2, points, C], every member identical to deterministic")

# trained noise: members differ; validation gives members_val members, reproducibly
with torch.no_grad():
    for n_, p_ in noise.named_parameters():
        if "embed_aux.2" in n_ or n_.endswith("slot"):
            p_.normal_(0, 0.2)
out_t = mm.Model.forward(make_self(make_cf(True), noise), model_params, batch)
p = out_t.physical[0]["CARRA2"][0]
assert not torch.allclose(p[0], p[1])
noise.eval()
s_eval = make_self(make_cf(True), noise); s_eval.training = False
v1 = mm.Model.forward(s_eval, model_params, batch).physical[0]["CARRA2"][0]
noise.train(); noise.begin_forward(); noise.eval()
v2 = mm.Model.forward(s_eval, model_params, batch).physical[0]["CARRA2"][0]
assert v1.shape[0] == 3 and torch.equal(v1, v2)
print("trained noise: members differ; validation has 3 members and repeats exactly")
print("ALL MODEL MEMBER TESTS PASSED")
