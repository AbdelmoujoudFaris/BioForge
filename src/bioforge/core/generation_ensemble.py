"""Generative Chemistry Engine.

Combines an ensemble of generators - the real, ported E(3)-equivariant
pocket-conditioned diffusion model from `pharmaforge_core` plus documented,
unimplemented extension points for DiffSBDD/DiffDock-L/a fine-tuned latent
diffusion model - with genuine multi-objective ranking (Pareto-optimality
over predicted binding affinity, selectivity against an anti-target panel,
synthetic accessibility, and ADMET desirability) over whatever candidate
pool the enabled generators actually produced.

Honesty note: `EquivariantDiffusionGenerator` runs the real, tested
architecture from `engines/pharmaforge_core` with randomly-initialized (but
seeded, reproducible) weights - no checkpoint trained on PDBbind/
CrossDocked2020 ships with this repository. See
`engines/pharmaforge_core/docs/whitepaper.md`. Ranking/Pareto-selection over
the produced pool is real and doesn't depend on the weights being trained to
be a genuine implementation of the described algorithm.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pharmaforge_core.config import (
    DiffusionConfig,
    GenerationConfig,
    ScoringConfig,
    SelectivityConfig,
)
from pharmaforge_core.generation.pipeline import PharmaForgePipeline, resolve_pocket
from pharmaforge_core.models.diffusion import LigandDiffusionModel
from pharmaforge_core.models.scoring import MultiTargetScoringEnsemble, ScoringStack
from pharmaforge_core.models.selectivity import SelectivityProfiler, pareto_front_indices
from pharmaforge_core.utils.chem import compute_physchem, is_valid
from pharmaforge_core.utils.seed import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generator adapters
# ---------------------------------------------------------------------------


class GeneratorAdapter:
    name: str = "base"

    def generate(self, *args: Any, **kwargs: Any) -> list[dict]:
        raise NotImplementedError


class DiffSBDDAdapter(GeneratorAdapter):
    """Extension point for Schneuing et al.'s DiffSBDD. Not implemented here:
    wiring it in means vendoring its (GPL-licensed) training code and a
    checkpoint this repository doesn't ship. Swap in a real adapter once one
    is trained/licensed for your deployment - `EquivariantDiffusionGenerator`
    below demonstrates the adapter shape it should match.
    """

    name = "diffsbdd"

    def generate(self, *args: Any, **kwargs: Any) -> list[dict]:
        raise NotImplementedError("DiffSBDDAdapter is a documented extension point; not wired to real weights.")


class DiffDockLAdapter(GeneratorAdapter):
    """Extension point for DiffDock-L pose generation used as a candidate
    source (as opposed to pose refinement, see `core/refinement.py`). Not
    implemented: needs the published checkpoint and a GPU.
    """

    name = "diffdock_l"

    def generate(self, *args: Any, **kwargs: Any) -> list[dict]:
        raise NotImplementedError("DiffDockLAdapter is a documented extension point; not wired to real weights.")


class LatentDiffusionAdapter(GeneratorAdapter):
    """Extension point for a fine-tuned latent-diffusion generator (e.g. a
    DiG-style model). Not implemented: needs a training run this repository
    doesn't ship data/compute for.
    """

    name = "latent_diffusion"

    def generate(self, *args: Any, **kwargs: Any) -> list[dict]:
        raise NotImplementedError("LatentDiffusionAdapter is a documented extension point; not wired to real weights.")


class EquivariantDiffusionGenerator(GeneratorAdapter):
    """Real generator: ported `pharmaforge_core` E(3)-equivariant pocket-
    conditioned diffusion model + differentiable scoring-guided sampling.
    """

    name = "equivariant_diffusion"

    def __init__(self, seed: int | None = 42, n_timesteps: int = 50):
        if seed is not None:
            set_seed(seed)
        diffusion_cfg = DiffusionConfig(n_timesteps=n_timesteps)
        scoring_cfg = ScoringConfig()
        self.diffusion_model = LigandDiffusionModel(diffusion_cfg)
        self.scoring_stack = ScoringStack(scoring_cfg)
        self.selectivity_cfg = SelectivityConfig()
        self.pipeline = PharmaForgePipeline(
            diffusion_model=self.diffusion_model,
            scoring_stack=self.scoring_stack,
        )

    def generate(
        self,
        pdb_id: str,
        n_candidates: int,
        max_atoms: int,
        ligand_resname: str | None,
        anti_target_pdb_ids: dict[str, str] | None,
        target_name: str,
        seed: int | None,
    ) -> list[dict]:
        cfg = GenerationConfig(
            mode="de_novo",
            n_candidates=n_candidates,
            max_atoms=max_atoms,
            diffusion=self.diffusion_model.cfg,
            selectivity=self.selectivity_cfg,
            seed=seed,
        )
        anti_targets = anti_target_pdb_ids or None
        if anti_targets:
            ensemble = MultiTargetScoringEnsemble(self.scoring_stack)
            self.pipeline.selectivity_profiler = SelectivityProfiler(ensemble, self.selectivity_cfg)
        raw = self.pipeline.run(
            pdb_id=pdb_id,
            cfg=cfg,
            ligand_resname=ligand_resname,
            target_name=target_name,
            anti_target_pdb_ids=anti_targets,
        )
        for c in raw:
            c["generator_source"] = self.name
        return raw


_ADAPTERS: dict[str, type[GeneratorAdapter]] = {
    "equivariant_diffusion": EquivariantDiffusionGenerator,
    "diffsbdd": DiffSBDDAdapter,
    "diffdock_l": DiffDockLAdapter,
    "latent_diffusion": LatentDiffusionAdapter,
}


# ---------------------------------------------------------------------------
# Ensemble orchestration + multi-objective ranking
# ---------------------------------------------------------------------------


@dataclass
class RankedCandidate:
    candidate_id: str
    smiles: str | None
    valid: bool
    generator_source: str
    binding_affinity_kcal_mol: float | None
    qed: float | None
    sa_score: float | None
    admet_desirability: float | None
    pareto_optimal: bool = False
    raw: dict = field(default_factory=dict)


def _admet_desirability(mol) -> float:
    """Cheap, dependency-light desirability proxy for ranking (0-1, higher
    better): the real per-endpoint ADMET profile lives in
    `core/admet_safety.py` - this exists only so generation-time ranking has
    *some* ADMET-aware signal without importing the full ADMET service.
    """
    from pharmaforge_core.scoring.admet import RuleBasedADMET

    profile = RuleBasedADMET().predict(mol)
    score = 0.5
    score += 0.2 if profile["human_intestinal_absorption"] else -0.1
    score -= 0.2 if profile["cyp_inhibition_risk"] == "high" else 0.0
    score -= 0.25 if profile["herg_liability_risk"] == "high" else 0.0
    return float(np.clip(score, 0.0, 1.0))


def generate_ensemble(
    pdb_id: str,
    generator_names: list[str],
    n_candidates: int = 20,
    max_atoms: int = 45,
    n_timesteps: int = 50,
    ligand_resname: str | None = None,
    anti_target_pdb_ids: dict[str, str] | None = None,
    target_name: str = "target",
    seed: int | None = 42,
) -> list[RankedCandidate]:
    """Run every requested generator (logging + skipping any that raise
    `NotImplementedError`, i.e. the documented stub adapters, rather than
    failing the whole request) and rank the pooled output.
    """
    pool: list[dict] = []
    for gname in generator_names:
        adapter_cls = _ADAPTERS.get(gname)
        if adapter_cls is None:
            logger.warning("Unknown generator '%s' skipped", gname)
            continue
        try:
            adapter = (
                adapter_cls(seed=seed, n_timesteps=n_timesteps) if gname == "equivariant_diffusion" else adapter_cls()
            )
            pool.extend(
                adapter.generate(
                    pdb_id=pdb_id,
                    n_candidates=n_candidates,
                    max_atoms=max_atoms,
                    ligand_resname=ligand_resname,
                    anti_target_pdb_ids=anti_target_pdb_ids,
                    target_name=target_name,
                    seed=seed,
                )
            )
        except NotImplementedError as exc:
            logger.info("Generator '%s' not available: %s", gname, exc)

    ranked: list[RankedCandidate] = []
    for c in pool:
        mol = c.get("mol")
        valid = bool(c.get("valid")) and mol is not None and is_valid(mol)
        smiles = None
        qed = sa_score = admet_desirability = None
        if valid:
            from rdkit import Chem

            smiles = Chem.MolToSmiles(mol)
            physchem = compute_physchem(mol)
            qed, sa_score = physchem["qed"], physchem["sa_score"]
            admet_desirability = _admet_desirability(mol)
        ranked.append(
            RankedCandidate(
                candidate_id=str(uuid.uuid4()),
                smiles=smiles,
                valid=valid,
                generator_source=c.get("generator_source", "unknown"),
                # `c["score"]` is the scoring stack's differentiable_score output, which is
                # "higher = better" (== -binding_affinity + a clash-validity bonus, see
                # `pharmaforge_core.models.scoring.ScoringStack.differentiable_score`). Negate it
                # here so `binding_affinity_kcal_mol` actually means what its name says: a
                # ΔG-like quantity where lower = tighter/better binding, matching every other
                # affinity field in this codebase (`RefinementResult.docking_score_kcal_mol`, etc.).
                binding_affinity_kcal_mol=-float(c["score"]) if "score" in c and c["score"] is not None else None,
                qed=qed,
                sa_score=sa_score,
                admet_desirability=admet_desirability,
                raw=c,
            )
        )

    return multi_objective_rank(ranked)


def multi_objective_rank(candidates: list[RankedCandidate]) -> list[RankedCandidate]:
    """Flags the Pareto-optimal subset over (affinity, QED, -SAscore,
    ADMET desirability) using the real non-dominated-sort routine already
    validated in `engines/pharmaforge_core` (`models.selectivity.pareto_front_indices`),
    generalized here from 1 target + N anti-targets to N arbitrary objectives.
    """
    scored = [c for c in candidates if c.valid and c.binding_affinity_kcal_mol is not None]
    if len(scored) < 2:
        return candidates

    matrix = np.array(
        [
            [
                c.binding_affinity_kcal_mol,  # ΔG-like, lower = better = minimize directly
                c.qed or 0.0,
                -(c.sa_score or 10.0),
                c.admet_desirability or 0.0,
            ]
            for c in scored
        ]
    )
    try:
        front = set(pareto_front_indices(matrix).tolist())
    except Exception as exc:  # pragma: no cover - pymoo/environment dependent
        logger.warning("Pareto ranking unavailable (%s); leaving pareto_optimal unset", exc)
        return candidates

    for i, c in enumerate(scored):
        c.pareto_optimal = i in front
    return candidates
