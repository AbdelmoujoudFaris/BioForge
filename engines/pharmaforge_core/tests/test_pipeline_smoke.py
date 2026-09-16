"""End-to-end smoke test: diffusion generation -> scoring -> mol reconstruction,
the same path exercised by `generation.sampler.GuidedDiffusionSampler` and by
the dashboard, using tiny models so it runs in well under a second on CPU.
"""
from pharmaforge_core.config import DiffusionConfig, GenerationConfig, ScoringConfig
from pharmaforge_core.generation.sampler import GuidedDiffusionSampler
from pharmaforge_core.models.diffusion import LigandDiffusionModel
from pharmaforge_core.models.scoring import ScoringStack


def test_guided_diffusion_sampler_end_to_end(small_egnn_config, toy_pocket):
    diffusion_cfg = DiffusionConfig(n_timesteps=4, max_ligand_atoms=5, egnn=small_egnn_config)
    scoring_cfg = ScoringConfig(egnn=small_egnn_config)

    diffusion_model = LigandDiffusionModel(diffusion_cfg)
    scoring_stack = ScoringStack(scoring_cfg)
    sampler = GuidedDiffusionSampler(diffusion_model, scoring_stack)

    gen_cfg = GenerationConfig(mode="de_novo", n_candidates=2, max_atoms=5, diffusion=diffusion_cfg, scoring=scoring_cfg)
    candidates = sampler.generate(toy_pocket, gen_cfg)

    assert len(candidates) == 2
    assert candidates[0].score >= candidates[1].score  # sorted descending by score
    for c in candidates:
        assert c.coords.shape == (5, 3)
        assert len(c.elements) == 5
