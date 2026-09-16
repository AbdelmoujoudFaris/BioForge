import torch

from pharmaforge_core.config import ActorCriticConfig, ScoringConfig
from pharmaforge_core.models.actor_critic import ActorCritic, DifferentiableTreeSearch, GraphTransformerActor
from pharmaforge_core.models.scoring import ScoringStack


def test_actor_sample_action_shapes(small_egnn_config, toy_pocket, toy_ligand):
    cfg = ActorCriticConfig(egnn=small_egnn_config, n_transformer_layers=1, n_heads=2)
    actor = GraphTransformerActor(cfg)
    action = actor.sample_action(toy_pocket, toy_ligand["atom_type_idx"], toy_ligand["physchem"], toy_ligand["coords"])

    assert action["new_atom_type"].dim() == 0
    assert action["attach_idx"].dim() == 0
    assert action["bond_type"].dim() == 0
    assert action["log_prob"].dim() == 0
    assert 0 <= action["attach_idx"].item() < toy_ligand["coords"].shape[0]


def test_actor_critic_advantage_computation(small_egnn_config, toy_pocket, toy_ligand):
    ac_cfg = ActorCriticConfig(egnn=small_egnn_config, n_transformer_layers=1, n_heads=2)
    critic = ScoringStack(ScoringConfig(egnn=small_egnn_config))
    actor_critic = ActorCritic(ac_cfg, critic)

    value = actor_critic.value(toy_pocket, toy_ligand["atom_type_idx"], toy_ligand["physchem"], toy_ligand["coords"])
    advantage = actor_critic.advantage(reward=torch.tensor(1.0), value=value, next_value=torch.tensor(0.0), done=True)
    assert torch.isclose(advantage, torch.tensor(1.0) - value)


def test_differentiable_tree_search_grows_molecule(small_egnn_config, toy_pocket, toy_ligand):
    ac_cfg = ActorCriticConfig(egnn=small_egnn_config, n_transformer_layers=1, n_heads=2)
    critic = ScoringStack(ScoringConfig(egnn=small_egnn_config))
    actor = GraphTransformerActor(ac_cfg)
    search = DifferentiableTreeSearch(actor, critic, beam_width=2, max_depth=2)

    beam = search.search(toy_pocket, toy_ligand["atom_type_idx"], toy_ligand["physchem"], toy_ligand["coords"])
    assert len(beam) <= 2
    for atom_type_idx, physchem, coords, score in beam:
        assert coords.shape[0] >= toy_ligand["coords"].shape[0]
        assert isinstance(score, float)
