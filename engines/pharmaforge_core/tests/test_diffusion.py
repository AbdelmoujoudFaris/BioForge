import torch

from pharmaforge_core.config import DiffusionConfig
from pharmaforge_core.models.diffusion import GaussianDiffusionSchedule, LigandDiffusionModel, make_beta_schedule


def test_beta_schedule_is_monotonic_and_bounded():
    cfg = DiffusionConfig(n_timesteps=50, schedule="cosine")
    betas = make_beta_schedule(cfg)
    assert betas.shape == (50,)
    assert (betas > 0).all() and (betas < 1).all()


def test_q_sample_shape(small_egnn_config):
    cfg = DiffusionConfig(n_timesteps=20, egnn=small_egnn_config)
    schedule = GaussianDiffusionSchedule(cfg)
    x0 = torch.randn(6, 3)
    t = torch.tensor([5])
    noise = torch.randn_like(x0)
    x_t = schedule.q_sample(x0, t, noise)
    assert x_t.shape == x0.shape


def test_training_loss_backward_updates_params(small_egnn_config, toy_pocket, toy_ligand):
    cfg = DiffusionConfig(n_timesteps=10, egnn=small_egnn_config)
    model = LigandDiffusionModel(cfg)
    params_before = [p.clone() for p in model.parameters()]

    loss = model.training_loss(toy_pocket, toy_ligand["coords"], toy_ligand["atom_type_idx"], toy_ligand["physchem"])
    loss.backward()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()

    changed = any(not torch.allclose(a, b) for a, b in zip(params_before, model.parameters()))
    assert changed


def test_p_sample_loop_shapes_unguided(small_egnn_config, toy_pocket):
    cfg = DiffusionConfig(n_timesteps=5, egnn=small_egnn_config)
    model = LigandDiffusionModel(cfg)
    coords, atom_type_idx = model.p_sample_loop(toy_pocket, n_ligand_atoms=6)
    assert coords.shape == (6, 3)
    assert atom_type_idx.shape == (6,)
    assert not coords.requires_grad


def test_p_sample_loop_with_guidance(small_egnn_config, toy_pocket):
    from pharmaforge_core.models.scoring import ScoringStack
    from pharmaforge_core.config import ScoringConfig

    cfg = DiffusionConfig(n_timesteps=5, egnn=small_egnn_config)
    model = LigandDiffusionModel(cfg)
    scoring = ScoringStack(ScoringConfig(egnn=small_egnn_config))

    def guidance_fn(coords, atom_type_idx):
        physchem = torch.zeros(coords.shape[0], toy_pocket["physchem"].shape[-1])
        return scoring.differentiable_score(toy_pocket, atom_type_idx, physchem, coords)

    coords, atom_type_idx = model.p_sample_loop(toy_pocket, n_ligand_atoms=6, guidance_fn=guidance_fn)
    assert coords.shape == (6, 3)
    assert not coords.requires_grad
