"""ADMET & Safety.

Layers, most-real-first:

  1. `RuleBasedADMET` (vendored, real) - BOILED-Egg-style descriptor
     heuristics for absorption/BBB/CYP/hERG. This is BioForge's first-class
     hERG liability flag, per the platform spec.
  2. `off_target_panel_screen` (real) - reuses the vendored, differentiable
     `ScoringStack`/`MultiTargetScoringEnsemble` to score a candidate against
     a *panel* of anti-target pockets, standing in for "proteome-wide
     docking" at a scale a request handler can actually run. Swap the panel
     for a larger curated anti-target set (kinome, hERG/Nav1.5/Cav1.2
     cardiac panel, CYP panel, ...) in a batch job for closer-to-proteome-wide
     coverage.
  3. `LearnedADMET` / `DeepTox` / `DTIAdapter` (vendored/new, documented
     stubs) - extension points for a trained multi-task ADMET model, a
     DeepTox-style toxicity classifier, and DeepConv-DTI/TransformerCPI-style
     off-target deep learning, none of which ship trained weights here (same
     honesty policy as `pharmaforge_core.scoring.admet.LearnedADMET`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rdkit import Chem

from pharmaforge_core.data.pdb_download import fetch_structure
from pharmaforge_core.data.pocket_extraction import extract_pocket_around_point, geometric_center_fallback
from pharmaforge_core.data.preprocessing import pocket_to_tensors
from pharmaforge_core.models.scoring import MultiTargetScoringEnsemble, ScoringStack
from pharmaforge_core.scoring.admet import RuleBasedADMET

# Re-exported so services/admet can offer the documented (unimplemented)
# extension point without importing the engine package directly.
from pharmaforge_core.scoring.admet import LearnedADMET  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass
class ADMETSafetyProfile:
    human_intestinal_absorption: bool | None
    blood_brain_barrier_permeant: bool | None
    cyp_inhibition_risk: str | None
    herg_liability_risk: str | None
    n_rotatable_bonds: int | None
    off_target_hits: list[dict] = field(default_factory=list)
    method: str = "rule_based_v1"


def profile_admet(mol: Chem.Mol) -> ADMETSafetyProfile:
    result = RuleBasedADMET().predict(mol)
    return ADMETSafetyProfile(
        human_intestinal_absorption=result["human_intestinal_absorption"],
        blood_brain_barrier_permeant=result["blood_brain_barrier_permeant"],
        cyp_inhibition_risk=result["cyp_inhibition_risk"],
        herg_liability_risk=result["herg_liability_risk"],
        n_rotatable_bonds=result["n_rotatable_bonds"],
        method=result["method"],
    )


class DeepToxAdapter:
    """Extension point for a DeepTox-style multi-task toxicity classifier
    (Mayr et al., 2016, Tox21 challenge winner). Not implemented: needs the
    Tox21/ToxCast training data and a dedicated training run.
    """

    def predict(self, mol: Chem.Mol) -> dict:  # pragma: no cover
        raise NotImplementedError("Wire a trained DeepTox-style Tox21/ToxCast classifier here.")


class DTIAdapter:
    """Extension point for DeepConv-DTI / TransformerCPI-style sequence-based
    drug-target interaction prediction. Not implemented: needs a pretrained
    checkpoint; `off_target_panel_screen` below is the real, structure-based
    alternative already wired up (differentiable scoring against a real
    anti-target pocket panel, not a stub).
    """

    def predict(self, smiles: str, target_sequence: str) -> dict:  # pragma: no cover
        raise NotImplementedError("Wire a trained DeepConv-DTI/TransformerCPI checkpoint here.")


DEFAULT_ANTI_TARGET_PANEL: dict[str, str] = {
    # a small, illustrative cardiac/liability panel - not exhaustive.
    "hERG_proxy": "5VA1",
}


def off_target_panel_screen(
    scoring_stack: ScoringStack,
    ligand_atom_type_idx,
    ligand_physchem,
    ligand_coords,
    panel_pdb_ids: dict[str, str] | None = None,
    flag_threshold: float = 0.0,
) -> list[dict]:
    """Score a candidate against each anti-target in `panel_pdb_ids` using
    the same differentiable scoring stack used for the primary target -
    real computation, small panel (a request-time-feasible stand-in for
    "proteome-wide" screening; run the same function over a larger panel as
    an offline batch job for broader coverage).
    """
    panel_pdb_ids = panel_pdb_ids or DEFAULT_ANTI_TARGET_PANEL
    ensemble = MultiTargetScoringEnsemble(scoring_stack)
    hits = []
    for name, pdb_id in panel_pdb_ids.items():
        try:
            pdb_path = fetch_structure(pdb_id)
            center = geometric_center_fallback(pdb_path)
            pocket = extract_pocket_around_point(pdb_path, center)
            pocket_tensors = pocket_to_tensors(pocket)
            score = ensemble.score(name, pocket_tensors, ligand_atom_type_idx, ligand_physchem, ligand_coords)
            score_val = float(score.item())
            hits.append(
                {
                    "anti_target": name,
                    "pdb_id": pdb_id,
                    "predicted_score": score_val,
                    "flagged": score_val > flag_threshold,
                    "method": "differentiable_scoring_panel",
                }
            )
        except Exception as exc:  # pragma: no cover - network/environment dependent
            logger.warning("Off-target screen against %s (%s) failed: %s", name, pdb_id, exc)
    return hits
