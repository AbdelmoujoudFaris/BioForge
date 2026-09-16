"""End-to-end generation pipeline: PDB ID / pocket in, ranked candidates out.

Dispatches to `GuidedDiffusionSampler` (de novo) or `FragmentGrower`
(fragment-growing / scaffold-hopping) based on `GenerationConfig.mode`, and
always returns the same `list[dict]` shape so `analysis/` and `ui/` don't
need to know which generation mode produced a given candidate.
"""
from __future__ import annotations

from pharmaforge_core.config import GenerationConfig
from pharmaforge_core.data.pdb_download import fetch_structure
from pharmaforge_core.data.pocket_extraction import (
    extract_pocket_around_ligand,
    extract_pocket_around_point,
    find_fpocket_cavities,
    geometric_center_fallback,
)
from pharmaforge_core.data.preprocessing import pocket_to_tensors
from pharmaforge_core.generation.fragment_growing import FragmentGrower, scaffold_from_reference, seed_from_smiles
from pharmaforge_core.generation.sampler import GuidedDiffusionSampler
from pharmaforge_core.models.actor_critic import ActorCritic
from pharmaforge_core.models.diffusion import LigandDiffusionModel
from pharmaforge_core.models.scoring import ScoringStack
from pharmaforge_core.models.selectivity import SelectivityProfiler


def resolve_pocket(pdb_id: str, ligand_resname: str | None, cfg: GenerationConfig):
    """PDB ID -> Pocket, using the reference ligand if one is named, else
    blind fpocket cavity detection with a geometric-center fallback.
    """
    pdb_path = fetch_structure(pdb_id)
    if ligand_resname:
        return extract_pocket_around_ligand(pdb_path, ligand_resname, cfg.pocket)

    centers = find_fpocket_cavities(pdb_path)
    center = centers[0] if centers else geometric_center_fallback(pdb_path)
    return extract_pocket_around_point(pdb_path, center, cfg.pocket)


class PharmaForgePipeline:
    def __init__(
        self,
        diffusion_model: LigandDiffusionModel,
        scoring_stack: ScoringStack,
        actor_critic: ActorCritic | None = None,
        selectivity_profiler: SelectivityProfiler | None = None,
        device: str = "cpu",
    ):
        self.diffusion_model = diffusion_model
        self.scoring_stack = scoring_stack
        self.actor_critic = actor_critic
        self.selectivity_profiler = selectivity_profiler
        self.device = device

    def run(
        self,
        pdb_id: str,
        cfg: GenerationConfig,
        ligand_resname: str | None = None,
        seed_smiles: str | None = None,
        target_name: str = "target",
        anti_target_pdb_ids: dict[str, str] | None = None,
    ) -> list[dict]:
        pocket = resolve_pocket(pdb_id, ligand_resname, cfg)
        pocket_tensors = pocket_to_tensors(pocket)

        anti_target_pockets = None
        if anti_target_pdb_ids:
            anti_target_pockets = {
                name: pocket_to_tensors(resolve_pocket(anti_pdb_id, None, cfg))
                for name, anti_pdb_id in anti_target_pdb_ids.items()
            }

        if cfg.mode == "de_novo":
            sampler = GuidedDiffusionSampler(
                self.diffusion_model, self.scoring_stack, self.selectivity_profiler, device=self.device
            )
            candidates = sampler.generate(
                pocket_tensors, cfg, target_name=target_name, anti_target_pockets=anti_target_pockets
            )
            return [
                {"elements": c.elements, "coords": c.coords, "score": c.score, "mol": c.mol_or_none, "valid": c.valid}
                for c in candidates
            ]

        if cfg.mode in ("fragment_growing", "scaffold_hopping"):
            if self.actor_critic is None:
                raise ValueError("fragment_growing/scaffold_hopping require an ActorCritic instance")
            if not seed_smiles:
                raise ValueError("fragment_growing/scaffold_hopping require `seed_smiles`")
            seed_mol = seed_from_smiles(seed_smiles)
            if cfg.mode == "scaffold_hopping":
                seed_mol = scaffold_from_reference(seed_mol)
            grower = FragmentGrower(self.actor_critic)
            return grower.grow(pocket_tensors, seed_mol, cfg)

        raise ValueError(f"Unknown generation mode: {cfg.mode}")
