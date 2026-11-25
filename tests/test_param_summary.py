# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""[PARAMS] parameter summary: grouping by parameter name, totals, trainable counts."""

import logging

from torch import nn

from weathergen.model.param_summary import classify, log_parameter_summary, parameter_table


def test_classify():
    assert classify("encoder.embed_engine.embeds.ERA5.w", 8) == ("encoder", "embedder ERA5")
    assert classify("encoder.ae_global_engine.ae_global_blocks.0.w", 8) == (
        "encoder",
        "hl8 global engine",
    )
    assert classify("encoder._ae_global_per_level.5.blocks.w", 8) == (
        "encoder",
        "hl5 global engine",
    )
    assert classify("module.encoder.latent_cascade._ops.x", 8) == ("encoder", "latent cascade")
    assert classify("forecast_cascade._ops.x", 8) == ("processor", "forecast cascade")
    assert classify("latent_adapter.head.proj_out.8.weight", 8)[0] == "adapter"
    assert classify("target_token_engines.CARRA2.tte.w", 8) == ("decoders", "CARRA2")
    assert classify("latent_heads.JEPA.blocks.0.w", 8) == ("ssl", "JEPA head")
    assert classify("something_new.w", 8) == ("other", "something_new")


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.healpix_level = 8
        self.encoder = nn.Module()
        self.encoder.ae_global_engine = nn.Linear(4, 4)  # 20
        self.pred_heads = nn.ModuleDict({"CARRA2": nn.Linear(4, 2)})  # 10
        self.target_token_engines = nn.ModuleDict({"CARRA2": nn.Linear(2, 2)})  # 6
        self.forecast_cascade = nn.Linear(3, 3)  # 12
        self.forecast_cascade.weight.requires_grad = False  # 9 frozen


def test_table_and_log(caplog):
    lines = parameter_table(_Toy())
    text = "\n".join(lines)
    assert "Model" in lines[1] and "48" in lines[1] and "39" in lines[1]
    assert "hl8 global engine" in text and "Decoders" in text
    assert "(token engine 6 / head 10)" in text
    with caplog.at_level(logging.INFO, logger="weathergen.model.param_summary"):
        log_parameter_summary(_Toy())
    assert caplog.text.count("[PARAMS]") >= 8
