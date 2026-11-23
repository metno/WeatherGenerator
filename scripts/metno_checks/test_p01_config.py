"""p01 pre-training config: load it like the launcher (default config + experiment config +
streams_directory) and build the masks of every stream for train and val (CPU, no data).

Usage:
    uv run --offline python scripts/metno_checks/test_p01_config.py <config.yml> [streams_dir]

Checks:
  - every loss term carries the same target_source_correspondence ("independent")
  - SYNOP: geoinfos / stats_file / drop_empty_rows / single file
  - masks: input = random part of the cells (rate of the stream), target = every cell,
    for every stream, in train and val
"""
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf, open_dict

from weathergen.common.config import (
    _load_base_conf,
    _load_overwrite_conf,
    _load_streams_in_config,
)
from weathergen.datasets.masking import Masker
from weathergen.train.utils import get_active_stage_config

cfg_path = Path(sys.argv[1])
over = _load_overwrite_conf(cfg_path)
if len(sys.argv) > 2:
    with open_dict(over):
        over.streams_directory = sys.argv[2]
over = _load_streams_in_config(over)
base = _load_base_conf(None)
with open_dict(base):
    base.streams = None
cf = OmegaConf.merge(base, over)

print(f"streams: {list(cf.streams.keys())}")

# losses: every term with the same correspondence
lf = cf.training_config.losses.physical.loss_fcts
corrs = {k: OmegaConf.to_container(v.get("target_source_correspondence")) for k, v in lf.items()}
for k, v in lf.items():
    print(f"  loss {k:42s} weight {v.weight:<6} corr {corrs[k]}")
assert all(c == {0: {0: "independent"}} for c in corrs.values()), corrs

# SYNOP stream settings
s = cf.streams.SYNOP
assert s.type == "station" and len(s.filenames) == 1, "SYNOP: one file (sizes come from file 0)"
assert "geoinfos" in s and "geoinfo_channels" not in s
assert s.get("stats_file") and s.get("drop_empty_rows") is True
print(f"SYNOP: {s.filenames[0]}\n       stats from {s.stats_file}\n       {len(s.geoinfos)} geoinfos")

# stream ids unique
ids = [v.stream_id for v in cf.streams.values()]
assert len(ids) == len(set(ids)), ids

# masks per stream (the loader fills <stage>_source/target_channels from the readers)
streams = OmegaConf.create(OmegaConf.to_container(cf.streams))
with open_dict(streams):
    for v in streams.values():
        for st in ["train", "val"]:
            v[f"{st}_source_channels"] = list(v.source)
            v[f"{st}_target_channels"] = list(v.target)

stage_cfgs = {
    "train": cf.training_config,
    "val": get_active_stage_config(cf.training_config, cf.validation_config, ["losses"]),
}
for stage, mode_cfg in stage_cfgs.items():
    m = Masker(5, stage, streams, mode_cfg)
    m.reset_rng(np.random.default_rng(0))
    for name, sc in streams.items():
        rate = sc.masking_override.model_input.masking_strategy_config.rate
        fs, ft = [], []
        for _ in range(40):
            tm, sm, _ = m.build_samples_for_stream(["masking"], m.healpix_num_cells, sc)
            fs.append(np.asarray(sm.masks[0]).mean())
            ft.append(np.asarray(tm.masks[0]).mean())
        print(f"  {stage:5s} {name:7s}: input {np.mean(fs):.2f} (rate {rate}), "
              f"target {np.min(ft):.2f}")
        assert abs(np.mean(fs) - rate) < 0.03 and np.min(ft) == 1.0
print("OK")
