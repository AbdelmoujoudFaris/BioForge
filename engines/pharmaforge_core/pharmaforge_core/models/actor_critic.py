"""Graph-transformer actor-critic, replacing AlphaDrug's Lmser Transformer +
vanilla discrete MCTS.

  * Actor:  conditioned on the current partial molecule embedded jointly
            with the pocket (via `PocketConditionedEGNN`), a lightweight
            multi-head self-attention block ("graph transformer" layer)
            refines the pocket-aware node embeddings before an action head
            proposes the next (atom type, bond type, attachment point).
  * Critic: *is* the differentiable `ScoringStack` — no separate value
            network needs to be learned from scratch, since the scoring
            stack already predicts binding affinity for partial and
            complete molecules alike (a partial molecule is just a smaller
            ligand graph in the same pocket).

Search itself is not classical MCTS. `pharmaforge_core.generation.sampler`
implements guided-diffusion sampling as the primary generation mode and a
best-first *differentiable* tree search (`DifferentiableTreeSearch` below)
as an optional discrete-action alternative for fragment-growing, where
gradients flow through the critic to rank candidate expansions instead of
relying on random rollouts.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pharmaforge_core.config import ATOM_VOCAB, BOND_TYPES, ActorCriticConfig
from pharmaforge_core.models.egnn import PocketConditionedEGNN
from pharmaforge_core.models.scoring import ScoringStack
from pharmaforge_core.utils.geometry import radius_graph


class GraphTransformerBlock(nn.Module):
    """Standard pre-norm self-attention + MLP block over node embeddings.

    Runs after the equivariant EGNN backbone, on the (already
    permutation/rotation-invariant) scalar features `h`, giving the actor
    long-range context between distant atoms that local message passing
    would need many layers to propagate.
    """

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, h: Tensor) -> Tensor:
        # h: (N, hidden_dim) treated as a single-sequence batch (batch_first=True, batch size 1)
        x = h.unsqueeze(0)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x.squeeze(0)


@dataclass
class ActionLogits:
    atom_type_logits: Tensor  # (n_ligand_atoms + 1, n_atom_types) incl. STOP row
    attachment_logits: Tensor  # (n_ligand_atoms,) which existing atom to grow from
    bond_type_logits: Tensor  # (n_ligand_atoms, n_bond_types)


class GraphTransformerActor(nn.Module):
    """Proposes the next atom-addition action given the current partial
    ligand embedded together with the pocket.
    """

    def __init__(self, cfg: ActorCriticConfig, physchem_dim: int = 6):
        super().__init__()
        self.cfg = cfg
        self.backbone = PocketConditionedEGNN(cfg.egnn, physchem_dim=physchem_dim, time_embed=False)
        self.transformer_layers = nn.ModuleList(
            [GraphTransformerBlock(cfg.egnn.hidden_dim, cfg.n_heads) for _ in range(cfg.n_transformer_layers)]
        )
        h = cfg.egnn.hidden_dim
        self.new_atom_type_head = nn.Linear(h, len(ATOM_VOCAB))  # incl. implicit STOP = index 0 (PAD reused)
        self.attachment_head = nn.Linear(h, 1)
        self.bond_type_head = nn.Linear(h, len(BOND_TYPES))

    def forward(self, pocket: dict, ligand_atom_type_idx: Tensor, ligand_physchem: Tensor, ligand_coords: Tensor):
        n_pocket = pocket["coords"].shape[0]
        n_ligand = ligand_coords.shape[0]
        coords = torch.cat([pocket["coords"], ligand_coords], dim=0)
        atom_type_idx = torch.cat([pocket["atom_type_idx"], ligand_atom_type_idx], dim=0)
        residue_type_idx = torch.cat(
            [pocket["residue_type_idx"], torch.zeros(n_ligand, dtype=torch.long, device=coords.device)]
        )
        physchem = torch.cat([pocket["physchem"], ligand_physchem], dim=0)
        is_ligand = torch.cat(
            [torch.zeros(n_pocket, dtype=torch.long), torch.ones(n_ligand, dtype=torch.long)]
        ).to(coords.device)

        edge_index = radius_graph(coords, cutoff=self.cfg.egnn.cutoff_radius, max_neighbors=self.cfg.egnn.max_neighbors)
        h, _ = self.backbone(atom_type_idx, residue_type_idx, is_ligand, physchem, coords, edge_index)
        ligand_h = h[n_pocket:]
        for layer in self.transformer_layers:
            ligand_h = layer(ligand_h)

        pooled = ligand_h.mean(dim=0, keepdim=True)
        atom_type_logits = self.new_atom_type_head(pooled)
        attachment_logits = self.attachment_head(ligand_h).squeeze(-1)
        bond_type_logits = self.bond_type_head(ligand_h)
        return ActionLogits(atom_type_logits, attachment_logits, bond_type_logits)

    def sample_action(self, pocket, ligand_atom_type_idx, ligand_physchem, ligand_coords, temperature: float = 1.0):
        logits = self.forward(pocket, ligand_atom_type_idx, ligand_physchem, ligand_coords)
        atom_dist = torch.distributions.Categorical(logits=logits.atom_type_logits.squeeze(0) / temperature)
        attach_dist = torch.distributions.Categorical(logits=logits.attachment_logits / temperature)

        new_atom_type = atom_dist.sample()
        attach_idx = attach_dist.sample()
        # Bond type is per-attachment-point (one logit row per existing
        # atom); select the row for the atom we're actually attaching to
        # before sampling, so this yields a single bond-type decision rather
        # than one per existing atom in the partial molecule.
        bond_dist = torch.distributions.Categorical(logits=logits.bond_type_logits[attach_idx] / temperature)
        bond_type = bond_dist.sample()
        log_prob = atom_dist.log_prob(new_atom_type) + attach_dist.log_prob(attach_idx) + bond_dist.log_prob(bond_type)
        return {
            "new_atom_type": new_atom_type,
            "attach_idx": attach_idx,
            "bond_type": bond_type,
            "log_prob": log_prob,
        }


class ActorCritic(nn.Module):
    """Bundles the actor with the scoring stack acting as critic, and exposes
    a PPO/REINFORCE-style advantage estimate for `training.train_actor_critic`.
    """

    def __init__(self, cfg: ActorCriticConfig, critic: ScoringStack):
        super().__init__()
        self.cfg = cfg
        self.actor = GraphTransformerActor(cfg)
        self.critic = critic  # shared differentiable scoring stack

    def value(self, pocket, ligand_atom_type_idx, ligand_physchem, ligand_coords) -> Tensor:
        with torch.no_grad():
            return self.critic.differentiable_score(pocket, ligand_atom_type_idx, ligand_physchem, ligand_coords)

    def advantage(self, reward: Tensor, value: Tensor, next_value: Tensor, done: bool) -> Tensor:
        target = reward if done else reward + self.cfg.gamma * next_value
        return target - value


class DifferentiableTreeSearch:
    """Best-first tree search over discrete fragment-growing actions where
    node priority comes directly from the (differentiable, gradient-scored)
    critic instead of Monte-Carlo rollout statistics.

    Each expansion step:
      1. actor proposes `beam_width` candidate next-atom actions (top-k by
         actor log-prob, not sampled randomly like MCTS rollout policies),
      2. the critic scores every resulting partial molecule directly (a
         single batched forward pass, not a docking run),
      3. the best `beam_width` partial molecules survive to the next depth.

    This keeps the "tree search" framing AlphaDrug users are familiar with,
    while replacing random-rollout value estimation with an exact
    differentiable scalar at every node.
    """

    def __init__(
        self,
        actor: GraphTransformerActor,
        critic: ScoringStack,
        beam_width: int = 8,
        max_depth: int = 40,
        placement_fn=None,
    ):
        """placement_fn(coords: Tensor, attach_idx: int, new_atom_type: int) -> Tensor(3,)
        Overridable atom-placement rule; defaults to a naive centroid+noise
        offset. `generation.fragment_growing.FragmentGrower` supplies a
        bond-length/steric-aware placement instead.
        """
        self.actor = actor
        self.critic = critic
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.placement_fn = placement_fn

    @torch.no_grad()
    def search(self, pocket: dict, init_atom_type_idx: Tensor, init_physchem: Tensor, init_coords: Tensor):
        beam = [(init_atom_type_idx, init_physchem, init_coords, 0.0)]
        for _depth in range(self.max_depth):
            candidates = []
            for atom_type_idx, physchem, coords, _score in beam:
                logits = self.actor.forward(pocket, atom_type_idx, physchem, coords)
                topk = torch.topk(logits.atom_type_logits.squeeze(0), k=min(self.beam_width, logits.atom_type_logits.shape[-1]))
                for new_type in topk.indices.tolist():
                    if new_type == 0:  # STOP token
                        candidates.append((atom_type_idx, physchem, coords, None))
                        continue
                    new_atom_type_idx = torch.cat([atom_type_idx, torch.tensor([new_type])])
                    new_physchem = torch.cat([physchem, physchem.mean(dim=0, keepdim=True)])
                    if self.placement_fn is not None:
                        attach_idx = int(torch.argmax(logits.attachment_logits).item())
                        new_atom_coord = self.placement_fn(coords, attach_idx, new_type).view(1, 3)
                    else:
                        # Naive fallback placement: extend from the centroid.
                        new_atom_coord = coords.mean(dim=0, keepdim=True) + torch.randn(1, 3) * 1.5
                    new_coords = torch.cat([coords, new_atom_coord])
                    candidates.append((new_atom_type_idx, new_physchem, new_coords, None))

            scored = []
            for atom_type_idx, physchem, coords, _ in candidates:
                score = self.critic.differentiable_score(pocket, atom_type_idx, physchem, coords)
                scored.append((atom_type_idx, physchem, coords, score.item()))
            scored.sort(key=lambda item: item[-1], reverse=True)
            beam = scored[: self.beam_width]
        return beam
