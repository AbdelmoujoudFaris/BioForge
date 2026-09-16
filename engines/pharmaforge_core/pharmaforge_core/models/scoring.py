"""Differentiable multi-task scoring stack: the surrogate that replaces
AlphaDrug's external SMINA docking calls inside the search loop.

AlphaDrug spends its entire MCTS rollout budget waiting on a real docking
program (seconds per pose, run thousands of times). Here, a single forward
pass (milliseconds, batched, GPU-resident, and differentiable w.r.t. atomic
coordinates) predicts everything the search loop and the guided-diffusion
sampler need:

  1. binding_affinity : predicted Vina-like ΔG (kcal/mol, lower = better)
  2. clash_logit       : pose-validity / steric-clash classifier
  3. interaction_fp    : per-pocket-residue interaction fingerprint over
                         {h-bond donor, h-bond acceptor, hydrophobic,
                          pi-stacking, halogen-bond, salt-bridge}

It shares the `PocketConditionedEGNN` backbone with the diffusion generator
and the actor-critic so that "how do I read a 3D pocket" is learned once.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from pharmaforge_core.config import ScoringConfig
from pharmaforge_core.models.egnn import PocketConditionedEGNN
from pharmaforge_core.utils.geometry import radius_graph


class ScoringStack(nn.Module):
    def __init__(self, cfg: ScoringConfig, physchem_dim: int = 6):
        super().__init__()
        self.cfg = cfg
        self.backbone = PocketConditionedEGNN(cfg.egnn, physchem_dim=physchem_dim, time_embed=False)
        h = cfg.egnn.hidden_dim

        self.affinity_head = nn.Sequential(
            nn.Linear(h, cfg.affinity_hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.affinity_hidden_dim, 1),
        )
        self.clash_head = nn.Sequential(
            nn.Linear(h, cfg.affinity_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.affinity_hidden_dim, 1),
        )
        self.interaction_head = nn.Sequential(
            nn.Linear(2 * h, cfg.affinity_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.affinity_hidden_dim, cfg.fingerprint_dim),
        )

    def encode(
        self,
        pocket: dict,
        ligand_atom_type_idx: Tensor,
        ligand_physchem: Tensor,
        ligand_coords: Tensor,
        cutoff: float = 10.0,
        max_neighbors: int = 32,
    ) -> tuple[Tensor, Tensor, int]:
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
        edge_index = radius_graph(coords, cutoff=cutoff, max_neighbors=max_neighbors)
        h, _ = self.backbone(atom_type_idx, residue_type_idx, is_ligand, physchem, coords, edge_index)
        return h, edge_index, n_pocket

    def forward(
        self,
        pocket: dict,
        ligand_atom_type_idx: Tensor,
        ligand_physchem: Tensor,
        ligand_coords: Tensor,
    ) -> dict:
        h, edge_index, n_pocket = self.encode(pocket, ligand_atom_type_idx, ligand_physchem, ligand_coords)
        pocket_h, ligand_h = h[:n_pocket], h[n_pocket:]

        pooled_ligand = ligand_h.mean(dim=0, keepdim=True)
        affinity = self.affinity_head(pooled_ligand).squeeze(-1)
        clash_logit = self.clash_head(ligand_h).mean(dim=0)

        # Per-pocket-residue interaction fingerprint: pool ligand features
        # against every pocket atom, giving one fingerprint vector per pocket
        # atom describing its interaction with the whole ligand.
        n_pocket_atoms, n_ligand_atoms = pocket_h.shape[0], ligand_h.shape[0]
        pocket_expand = pocket_h.unsqueeze(1).expand(-1, n_ligand_atoms, -1)
        ligand_expand = ligand_h.unsqueeze(0).expand(n_pocket_atoms, -1, -1)
        pair_feats = torch.cat([pocket_expand, ligand_expand], dim=-1)
        interaction_logits = self.interaction_head(pair_feats)  # (n_pocket, n_ligand, fp_dim)

        return {
            "binding_affinity": affinity,               # (1,) kcal/mol-like scalar, lower is tighter
            "clash_logit": clash_logit,                  # (1,) higher = more likely a valid pose
            "interaction_fingerprint": interaction_logits,  # (n_pocket, n_ligand, fp_dim) raw logits
        }

    def differentiable_score(
        self,
        pocket: dict,
        ligand_atom_type_idx: Tensor,
        ligand_physchem: Tensor,
        ligand_coords: Tensor,
        clash_weight: float = 0.5,
    ) -> Tensor:
        """Single scalar combining affinity + validity, used directly as the
        guidance signal inside `LigandDiffusionModel.p_sample_loop` and as the
        critic value inside the actor-critic search.
        """
        out = self.forward(pocket, ligand_atom_type_idx, ligand_physchem, ligand_coords)
        # Affinity is a ΔG-like quantity (more negative = better); flip sign
        # so "higher score = better molecule" throughout the rest of the stack.
        score = -out["binding_affinity"] + clash_weight * torch.sigmoid(out["clash_logit"])
        return score.squeeze()


class MultiTargetScoringEnsemble(nn.Module):
    """Thin wrapper holding one `ScoringStack` per target/anti-target so the
    Selectivity Profiler (models/selectivity.py) can query binding predictions
    against an arbitrary panel of proteins without retraining anything.

    In production each entry would be a copy of the same architecture
    fine-tuned (or few-shot conditioned) on that target's pocket; here the
    ensemble simply dispatches by name and lets a single shared-weight model
    stand in when a target-specific checkpoint isn't available.
    """

    def __init__(self, shared: ScoringStack, target_specific: dict[str, ScoringStack] | None = None):
        super().__init__()
        self.shared = shared
        self.target_specific = nn.ModuleDict(target_specific or {})

    def score(self, target_name: str, pocket: dict, ligand_atom_type_idx, ligand_physchem, ligand_coords) -> Tensor:
        # nn.ModuleDict has no dict-style `.get`; check membership explicitly.
        model = self.target_specific[target_name] if target_name in self.target_specific else self.shared
        return model.differentiable_score(pocket, ligand_atom_type_idx, ligand_physchem, ligand_coords)
