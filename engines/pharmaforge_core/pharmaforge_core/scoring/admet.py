"""ADMET (absorption, distribution, metabolism, excretion, toxicity) profiling.

Ships two tiers:

  1. `RuleBasedADMET` - fast, dependency-free, descriptor-based heuristics
     (BOILED-Egg-style absorption/BBB estimate, a CYP3A4/2D6 inhibition risk
     flag, an hERG-liability flag) that always run, since they only need
     RDKit descriptors already computed for QED/SA/Lipinski.

  2. `LearnedADMET` - an optional plug-in point for a real trained ADMET
     model (e.g. a fine-tuned admet-ai / Chemprop checkpoint). Left as an
     interface with a clear `NotImplementedError` rather than faked numbers,
     since shipping a specific pretrained ADMET model is outside what this
     framework can validate without the corresponding training data license.

Both return the same flat dict shape so `analysis/physchem.py` and the
dashboard don't need to special-case which tier produced a given profile.
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors


class RuleBasedADMET:
    def predict(self, mol: Chem.Mol) -> dict:
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        mw = Descriptors.MolWt(mol)
        n_aromatic_rings = Descriptors.NumAromaticRings(mol)
        n_rotatable = Descriptors.NumRotatableBonds(mol)

        # BOILED-Egg-style ellipse heuristic (Daina & Zoetendal, 2016):
        # good passive human GI absorption for TPSA <~ 142 A^2 and LogP in a
        # sensible range; approximated here as an axis-aligned ellipse test
        # rather than fitting the original PCA ellipse boundary.
        hia = (tpsa <= 142.0) and (-2.0 <= logp <= 6.5)
        # BBB permeant heuristic: substantially stricter TPSA/LogP window.
        bbb = (tpsa <= 90.0) and (0.4 <= logp <= 6.0) and mw <= 450

        cyp_risk = "high" if (logp > 4.0 and n_aromatic_rings >= 2) else "low"
        herg_risk = "high" if (logp > 3.5 and n_aromatic_rings >= 1 and mw > 350) else "low"

        return {
            "human_intestinal_absorption": bool(hia),
            "blood_brain_barrier_permeant": bool(bbb),
            "cyp_inhibition_risk": cyp_risk,
            "herg_liability_risk": herg_risk,
            "n_rotatable_bonds": n_rotatable,
            "method": "rule_based_v1",
        }


class LearnedADMET:
    """Interface for a real trained multi-task ADMET model.

    Not implemented in this repository: a credible absorption/metabolism/
    toxicity model needs a licensed training set (e.g. Therapeutics Data
    Commons ADMET benchmarks) and a dedicated training run this project
    doesn't ship pretrained weights for. Wire a checkpoint here once one has
    been trained with `training/` conventions; until then callers should use
    `RuleBasedADMET`, which is honest about being a heuristic.
    """

    def __init__(self, checkpoint_path: str):
        raise NotImplementedError(
            "LearnedADMET requires a trained checkpoint (see docstring); use RuleBasedADMET for now."
        )

    def predict(self, mol: Chem.Mol) -> dict:  # pragma: no cover
        raise NotImplementedError
