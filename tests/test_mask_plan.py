# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests of the per-sample mask plan (datasets/mask_plan.py) on a regional hl5 + hl8 latent."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from weathergen.datasets.domain_pyramid import DomainPyramid
from weathergen.datasets.mask_plan import MaskPlanner, get_stream_roles, uses_mask_plan

# Scandinavia-like box
BBOX = {"lon_min": 0.0, "lon_max": 30.0, "lat_min": 55.0, "lat_max": 72.0}


@pytest.fixture(scope="module")
def pyramid():
    cf = OmegaConf.create(
        {"latent_levels": [{"level": 5, "domain": BBOX}, {"level": 8, "domain": BBOX}]}
    )
    return DomainPyramid.from_config(cf)


def make_planner(pyramid, strategies, roles=None, **extra):
    cfg = OmegaConf.create({"strategies": strategies, **extra})
    return MaskPlanner(pyramid, cfg, roles or {"era5": "both", "seviri": "both"})


def test_follow_policy_hides_children(pyramid):
    planner = make_planner(
        pyramid,
        {"joint": {"levels": {5: {"pattern": "blocks", "hidden": 0.5}, 8: "follow:5"}}},
    )
    planner.start_item(np.random.default_rng(0))
    plan = planner.get_plan(0)
    k5, k8 = plan.student[5].numpy(), plan.student[8].numpy()
    hidden5 = (~k5).mean()
    assert 0.4 <= hidden5 <= 0.75, hidden5
    anc = pyramid.ancestor_of(8, 5)
    inside = anc >= 0
    # hl8 cell hidden <=> its hl5 parent hidden
    assert np.array_equal(k8[inside], k5[anc[inside]])
    # the teacher sees everything
    assert plan.teacher[5].all() and plan.teacher[8].all()


def test_fine_only_and_unit_level(pyramid):
    planner = make_planner(
        pyramid,
        {
            "fine_only": {
                "levels": {
                    5: "visible",
                    8: {"pattern": "blocks", "unit_level": 5, "hidden": [0.3, 0.3]},
                }
            }
        },
    )
    planner.start_item(np.random.default_rng(1))
    plan = planner.get_plan(0)
    assert plan.student[5].all()
    k8 = plan.student[8].numpy()
    assert 0.0 < (~k8).mean() < 1.0
    # hidden in whole hl5 units: all children of a unit share the same state
    anc = pyramid.ancestor_of(8, 5)
    for u in np.unique(anc[anc >= 0]):
        vals = k8[anc == u]
        assert vals.all() or (~vals).all()


def test_uncorrelated_random_and_crop(pyramid):
    planner = make_planner(
        pyramid,
        {
            "unc": {
                "levels": {
                    5: {"pattern": "crop", "hidden": 0.5},
                    8: {"pattern": "random", "hidden": 0.75},
                }
            }
        },
    )
    planner.start_item(np.random.default_rng(2))
    plan = planner.get_plan(0)
    n8 = len(pyramid.domain(8))
    assert abs((~plan.student[8].numpy()).sum() - round(0.75 * n8)) <= 1
    n5 = len(pyramid.domain(5))
    assert abs((~plan.student[5].numpy()).sum() - round(0.5 * n5)) <= 1


def test_drop_and_mixture_frequencies(pyramid):
    planner = make_planner(
        pyramid,
        {
            "a": {"prob": 0.75, "levels": {5: "drop", 8: "visible"}},
            "b": {"prob": 0.25, "levels": {5: "visible", 8: "drop"}},
        },
    )
    rng = np.random.default_rng(3)
    counts = {"a": 0, "b": 0}
    for _ in range(2000):
        planner.start_item(rng)
        plan = planner.get_plan(0)
        counts[plan.strategy] += 1
        if plan.strategy == "a":
            assert not plan.student[5].any() and plan.student[8].all()
    assert 0.70 < counts["a"] / 2000 < 0.80


def test_shared_per_item_and_roles(pyramid):
    planner = make_planner(
        pyramid,
        {"joint": {"levels": {5: {"pattern": "blocks", "hidden": 0.5}, 8: "follow:5"}}},
        roles={"era5": "student", "seviri": "student", "carra2": "teacher"},
    )
    planner.start_item(np.random.default_rng(4))
    m_era5, p1 = planner.student_mask(0, 5, "era5")
    m_sev, p2 = planner.student_mask(0, 8, "seviri")
    m_carra_s, _ = planner.student_mask(0, 8, "carra2")
    # same plan object for all streams of the item
    assert p1["mask_plan"]["student"][8] is p2["mask_plan"]["student"][8]
    assert torch.equal(m_sev, p1["mask_plan"]["student"][8])
    # teacher-only stream never reaches the student, student-only never the teacher
    assert not m_carra_s.any()
    t_carra, _ = planner.teacher_mask(0, 8, "carra2")
    t_era5, _ = planner.teacher_mask(0, 5, "era5")
    assert t_carra.all() and not t_era5.any()
    # new item -> new plan
    planner.start_item(np.random.default_rng(5))
    m_era5_2, _ = planner.student_mask(0, 5, "era5")
    assert not torch.equal(m_era5, m_era5_2) or m_era5.all()


def test_config_validation(pyramid):
    with pytest.raises(AssertionError):
        make_planner(pyramid, {"x": {"levels": {5: "follow:8", 8: "visible"}}})
    with pytest.raises(AssertionError):
        make_planner(pyramid, {"x": {"levels": {7: "visible"}}})
    with pytest.raises(AssertionError):
        get_stream_roles({"s": {"ssl_view": "nobody"}})
    mode_cfg = OmegaConf.create({"model_input": {"s": {"masking_strategy": "mask_plan"}}})
    assert uses_mask_plan(mode_cfg)
