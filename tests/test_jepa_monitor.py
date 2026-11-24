import logging

import torch
from omegaconf import OmegaConf

from weathergen.train.jepa_monitor import JepaMonitor, representation_stats


def _cf(**mon):
    return OmegaConf.create(
        {
            "general": {"run_id": "test", "istep": 0},
            "train_logging": {"terminal": 10, "jepa_monitor": mon},
        }
    )


def test_representation_stats_perfect_and_collapsed():
    torch.manual_seed(0)
    t = torch.randn(2, 50, 16)
    m = torch.ones(2, 50, dtype=torch.bool)
    st = representation_stats(t, t, m)
    assert st["l1"] == 0.0 and st["skill"] == 0.0
    assert abs(st["cos"] - 1.0) < 1e-5 and st["tgt_std"] > 0.5 and 0 < st["erank"] <= 1
    # collapsed targets: every token the same vector
    c = torch.ones(2, 50, 16)
    st = representation_stats(t, c, m)
    assert st["tgt_std"] == 0.0
    # predicting the mean target: skill ~ 1
    st = representation_stats(t.mean(dim=(0, 1), keepdim=True).expand_as(t), t, m, erank=False)
    assert abs(st["skill"] - 1.0) < 0.05 and "erank" not in st
    # no selected token
    assert representation_stats(t, t, torch.zeros_like(m)) is None


def test_loss_breakdown_and_warnings(caplog):
    mon = JepaMonitor(_cf(every=5))
    assert mon.every == 5
    losses = {
        "LossLatentSSLStudentTeacher.JEPA": [0.5, 0.7],
        "LossLatentSSLStudentTeacher.JEPA_hl8": [0.6],
        "LossLatentSSLStudentTeacher.frac_hidden_hl8": [0.4],
        "LossLatentSSLStudentTeacher.context_hl5": [float("nan")],
        "LossLatentSSLStudentTeacher.loss_avg": [1.0],
        "LossPhysical.CARRA2.mse": [2.0],
    }
    with caplog.at_level(logging.INFO, logger="weathergen.train.jepa_monitor"):
        mon.log_loss_breakdown("train", 10, losses)
    text = caplog.text
    assert "[JEPA-MON] train istep=10 [LossLatentSSLStudentTeacher] JEPA=0.6" in text
    assert "JEPA_hl8=0.6" in text and "frac_hidden_hl8=0.4" in text
    assert "mse" not in text and "loss_avg" not in text
    assert "[JEPA-MON][WARN]" in text and "context_hl5" in text


def test_disabled_and_checksums():
    mon = JepaMonitor(_cf(enabled=False))
    assert mon.log_loss_breakdown("train", 0, {"x.JEPA": [1.0]}) is None
    lin = torch.nn.Linear(3, 2)
    ref = JepaMonitor._checksums(lin.named_parameters())
    assert set(ref) == {"weight", "bias"}
    with torch.no_grad():
        lin.weight.add_(1.0)
    now = JepaMonitor._checksums(lin.named_parameters())
    assert now["weight"] != ref["weight"] and now["bias"] == ref["bias"]
