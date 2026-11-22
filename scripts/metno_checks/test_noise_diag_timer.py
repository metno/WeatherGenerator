"""[diag noise] line and [timing] line (CPU)."""
import time
import torch
from weathergen.model import encoder_diag as ed
from weathergen.utils import step_timer as st


class Cf(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)

    def __getattr__(self, k):
        return self[k]


cf = Cf(encoder_diag=True, encoder_diag_every=1, ae_global_dim_embed=8, timing_diag=True,
        timing_diag_every=1)
ed._DIAG = None
d = ed.setup_encoder_diag(cf, [5, 8], {"ERA5": 5, "SEVIRI": 8})
d.begin_step(0, training=True)
x = torch.randn(1, 100, 8)
d.noise(8, x, x + 0.1 * torch.randn_like(x), 0, 100)
d.noise(8, x, x + 0.1 * torch.randn_like(x), 1, 100)
lines = d.report("train", 0)
ln = [l for l in lines if "[diag noise]" in l]
assert ln and "change by noise block 0.1" in ln[0] and "between members 0.14" in ln[0], lines
t = st.setup_step_timer(cf)
t.begin(0)
time.sleep(0.05); t.mark("encoder")
for _ in range(2):
    time.sleep(0.02); t.mark("members")
line = t.end()
assert "encoder 0.0" in line and "members 2 x 0.02" in line, line
print(line)
print("NOISE DIAG + TIMER TEST PASSED")
