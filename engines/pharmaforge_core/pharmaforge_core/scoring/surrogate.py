"""Inference-time convenience wrapper around `models.scoring.ScoringStack`.

Separated from `models/scoring.py` (which defines the architecture and is
trained by `training/train_scoring.py`) so that the analysis suite and the
dashboard can load a checkpoint and call `.evaluate(mol, pocket)` without
importing training-only concerns.
"""
from __future__ import annotations

from pathlib import Path

import torch
from rdkit import Chem

from pharmaforge_core.config import ScoringConfig
from pharmaforge_core.data.pocket_extraction import Pocket
from pharmaforge_core.data.preprocessing import ligand_mol_to_tensors, pocket_to_tensors
from pharmaforge_core.models.scoring import ScoringStack

INTERACTION_FP_LABELS = ["h_bond_donor", "h_bond_acceptor", "hydrophobic", "pi_stacking", "halogen_bond", "salt_bridge"]


class BindingSurrogate:
    """Loads a trained `ScoringStack` checkpoint and exposes a simple
    mol-in / dict-out evaluation API used everywhere outside of training.
    """

    def __init__(self, cfg: ScoringConfig, checkpoint_path: str | Path | None = None, device: str = "cpu"):
        self.device = device
        self.model = ScoringStack(cfg).to(device)
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, mol: Chem.Mol, pocket: Pocket) -> dict:
        pocket_tensors = pocket_to_tensors(pocket, device=self.device)
        ligand_tensors = ligand_mol_to_tensors(mol, device=self.device)
        out = self.model(
            pocket_tensors, ligand_tensors["atom_type_idx"], ligand_tensors["physchem"], ligand_tensors["coords"]
        )
        interaction_probs = torch.sigmoid(out["interaction_fingerprint"])
        per_type = interaction_probs.sum(dim=(0, 1))  # aggregate over pocket x ligand atom pairs
        return {
            "predicted_delta_g": -out["binding_affinity"].item(),
            "pose_validity": torch.sigmoid(out["clash_logit"]).item(),
            "interaction_fingerprint": {
                label: per_type[i].item() for i, label in enumerate(INTERACTION_FP_LABELS[: per_type.shape[0]])
            },
        }

    @torch.no_grad()
    def batch_evaluate(self, mols: list[Chem.Mol], pocket: Pocket) -> list[dict]:
        return [self.evaluate(mol, pocket) for mol in mols]
