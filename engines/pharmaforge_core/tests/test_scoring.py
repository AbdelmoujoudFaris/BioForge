import torch

from pharmaforge_core.config import ScoringConfig
from pharmaforge_core.models.scoring import MultiTargetScoringEnsemble, ScoringStack


def test_scoring_stack_output_shapes(small_egnn_config, toy_pocket, toy_ligand):
    cfg = ScoringConfig(egnn=small_egnn_config)
    model = ScoringStack(cfg, physchem_dim=6)
    out = model(toy_pocket, toy_ligand["atom_type_idx"], toy_ligand["physchem"], toy_ligand["coords"])

    n_pocket = toy_pocket["coords"].shape[0]
    n_ligand = toy_ligand["coords"].shape[0]
    assert out["binding_affinity"].shape == (1,)
    assert out["clash_logit"].shape == (1,)
    assert out["interaction_fingerprint"].shape == (n_pocket, n_ligand, cfg.fingerprint_dim)


def test_differentiable_score_has_gradient(small_egnn_config, toy_pocket, toy_ligand):
    cfg = ScoringConfig(egnn=small_egnn_config)
    model = ScoringStack(cfg, physchem_dim=6)
    coords = toy_ligand["coords"].clone().requires_grad_(True)

    score = model.differentiable_score(toy_pocket, toy_ligand["atom_type_idx"], toy_ligand["physchem"], coords)
    score.backward()

    assert coords.grad is not None
    assert coords.grad.abs().sum().item() > 0


def test_multi_target_ensemble_dispatches_by_name(small_egnn_config, toy_pocket, toy_ligand):
    cfg = ScoringConfig(egnn=small_egnn_config)
    shared = ScoringStack(cfg, physchem_dim=6)
    ensemble = MultiTargetScoringEnsemble(shared)

    score = ensemble.score("any_target_name", toy_pocket, toy_ligand["atom_type_idx"], toy_ligand["physchem"], toy_ligand["coords"])
    assert score.dim() == 0 or score.shape == (1,)
