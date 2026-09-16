"""Guided-diffusion sampling: wires the scoring stack (and, optionally, the
Selectivity Profiler) into `LigandDiffusionModel.p_sample_loop` as a
classifier-style guidance gradient, and turns the resulting raw point cloud
into an RDKit `Mol` plus its scoring-stack evaluation.

This is the primary generation mode described in the architecture spec
("Guided Diffusion Sampling instead of discrete MCTS, enabling end-to-end
gradient flow"). `DifferentiableTreeSearch` (models/actor_critic.py) is the
discrete-action alternative used by `fragment_growing.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pharmaforge_core.config import ATOM_VOCAB, GenerationConfig
from pharmaforge_core.models.diffusion import LigandDiffusionModel
from pharmaforge_core.models.scoring import ScoringStack
from pharmaforge_core.models.selectivity import SelectivityProfiler
from pharmaforge_core.utils.chem import is_valid, mol_from_coords


@dataclass
class Candidate:
    mol_or_none: object
    elements: list[str]
    coords: Tensor
    score: float
    valid: bool


def _atom_type_idx_to_elements(atom_type_idx: Tensor) -> list[str]:
    return [ATOM_VOCAB[i] if 0 <= i < len(ATOM_VOCAB) else "C" for i in atom_type_idx.tolist()]


class GuidedDiffusionSampler:
    """Samples molecules for a fixed pocket, optionally constrained by a
    Selectivity Profiler against anti-targets.
    """

    def __init__(
        self,
        diffusion_model: LigandDiffusionModel,
        scoring_stack: ScoringStack,
        selectivity_profiler: SelectivityProfiler | None = None,
        device: str | torch.device = "cpu",
    ):
        self.diffusion_model = diffusion_model.to(device)
        self.scoring_stack = scoring_stack.to(device)
        self.selectivity_profiler = selectivity_profiler
        self.device = device

    def _guidance_fn(self, pocket: dict, anti_target_pockets: dict[str, dict] | None, target_name: str):
        if self.selectivity_profiler is not None and anti_target_pockets:
            pockets = {target_name: pocket, **anti_target_pockets}

            def guided(coords, atom_type_idx):
                physchem = torch.zeros(coords.shape[0], pocket["physchem"].shape[-1], device=coords.device)
                return self.selectivity_profiler.constrained_guidance_score(
                    target_name, list(anti_target_pockets.keys()), pockets, atom_type_idx, physchem, coords
                )

            return guided

        def unconstrained(coords, atom_type_idx):
            physchem = torch.zeros(coords.shape[0], pocket["physchem"].shape[-1], device=coords.device)
            return self.scoring_stack.differentiable_score(pocket, atom_type_idx, physchem, coords)

        return unconstrained

    def generate(
        self,
        pocket: dict,
        cfg: GenerationConfig,
        target_name: str = "target",
        anti_target_pockets: dict[str, dict] | None = None,
    ) -> list[Candidate]:
        guidance_fn = self._guidance_fn(pocket, anti_target_pockets, target_name)
        candidates = []
        for _ in range(cfg.n_candidates):
            coords, atom_type_idx = self.diffusion_model.p_sample_loop(
                pocket,
                n_ligand_atoms=cfg.max_atoms,
                guidance_fn=guidance_fn,
                guidance_scale=cfg.diffusion.guidance_scale,
                device=self.device,
            )
            elements = _atom_type_idx_to_elements(atom_type_idx)
            mol = mol_from_coords(elements, coords.cpu().numpy())
            physchem = torch.zeros(coords.shape[0], pocket["physchem"].shape[-1], device=coords.device)
            score = self.scoring_stack.differentiable_score(pocket, atom_type_idx, physchem, coords).item()
            candidates.append(
                Candidate(mol_or_none=mol, elements=elements, coords=coords, score=score, valid=is_valid(mol))
            )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates
