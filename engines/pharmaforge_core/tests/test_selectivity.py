import numpy as np
import pytest
import torch

from pharmaforge_core.config import ScoringConfig, SelectivityConfig
from pharmaforge_core.models.scoring import MultiTargetScoringEnsemble, ScoringStack
from pharmaforge_core.models.selectivity import (
    SelectivityProfile,
    SelectivityProfiler,
    delta_g_to_kd,
    pareto_front_indices,
    run_multi_objective_search,
    selectivity_index,
)


def test_delta_g_to_kd_more_negative_means_tighter():
    assert delta_g_to_kd(-12.0) < delta_g_to_kd(-6.0)


def test_selectivity_index_selective_case():
    si = selectivity_index(delta_g_target=-10.0, delta_g_offtarget=-4.0)
    assert si > 1.0


def test_selectivity_index_nonselective_case():
    si = selectivity_index(delta_g_target=-8.0, delta_g_offtarget=-8.0)
    assert si == pytest.approx(1.0, rel=1e-6)


def test_selectivity_profile_worst_case():
    profile = SelectivityProfile(-9.0, {"a": -5.0, "b": -8.5})
    assert profile.worst_case_si == pytest.approx(min(profile.selectivity_index.values()))
    assert profile.selectivity_index["b"] < profile.selectivity_index["a"]  # b is a closer off-target call


def test_multi_objective_search_returns_pareto_front():
    rng = np.random.default_rng(0)
    scores = rng.uniform(-10, -4, size=(20, 3))
    cfg = SelectivityConfig(population_size=32, n_generations=10)
    result = run_multi_objective_search(scores, cfg, budget=4)
    assert result.F is not None
    assert result.F.shape[1] == 3


def test_selectivity_profiler_wiring(small_egnn_config, toy_pocket, toy_ligand):
    scoring = ScoringStack(ScoringConfig(egnn=small_egnn_config))
    ensemble = MultiTargetScoringEnsemble(scoring)
    profiler = SelectivityProfiler(ensemble, SelectivityConfig())

    anti_pocket = {k: v.clone() for k, v in toy_pocket.items()}
    pockets = {"target": toy_pocket, "anti_a": anti_pocket}

    profile = profiler.profile(
        "target", ["anti_a"], pockets, toy_ligand["atom_type_idx"], toy_ligand["physchem"], toy_ligand["coords"]
    )
    assert isinstance(profile, SelectivityProfile)
    assert "anti_a" in profile.selectivity_index

    guidance = profiler.constrained_guidance_score(
        "target", ["anti_a"], pockets, toy_ligand["atom_type_idx"], toy_ligand["physchem"], toy_ligand["coords"]
    )
    assert guidance.dim() == 0


def test_pareto_front_indices_direct_sort():
    # Candidate 0 dominates candidate 1 on every objective (tighter primary,
    # weaker on both anti-targets) -> candidate 1 must not appear on the front.
    scores = np.array(
        [
            [-9.0, -3.0, -3.0],
            [-8.0, -4.0, -4.0],
            [-7.0, -8.0, -2.0],
        ]
    )
    front = pareto_front_indices(scores)
    assert 1 not in front
    assert 0 in front
