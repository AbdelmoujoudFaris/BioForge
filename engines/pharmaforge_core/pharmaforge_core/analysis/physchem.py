"""Physicochemical & ADMET profiling for a batch of generated candidates."""
from __future__ import annotations

import pandas as pd
from rdkit import Chem

from pharmaforge_core.scoring.admet import RuleBasedADMET
from pharmaforge_core.scoring.retrosynthesis import RetrosynthesisScorer
from pharmaforge_core.utils.chem import compute_physchem, lipinski_pass


def profile_candidates(
    mols: list[Chem.Mol],
    admet: RuleBasedADMET | None = None,
    retro: RetrosynthesisScorer | None = None,
) -> pd.DataFrame:
    """One row per candidate: QED, SAscore, LogP, MW, Lipinski pass/fail,
    ADMET flags, and (if `retro` is given) a retrosynthesis-based
    synthetic-accessibility verdict.
    """
    admet = admet or RuleBasedADMET()
    rows = []
    for i, mol in enumerate(mols):
        if mol is None:
            rows.append({"candidate_id": i, "valid": False})
            continue
        props = compute_physchem(mol)
        row = {
            "candidate_id": i,
            "smiles": Chem.MolToSmiles(mol),
            "valid": True,
            **props,
            "lipinski_pass": lipinski_pass(props),
            **{f"admet_{k}": v for k, v in admet.predict(mol).items()},
        }
        if retro is not None:
            row.update({f"retro_{k}": v for k, v in retro.score(mol).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    valid = df[df.get("valid", False) == True]
    if valid.empty:
        return {"n_candidates": len(df), "n_valid": 0}
    return {
        "n_candidates": len(df),
        "n_valid": len(valid),
        "validity_rate": len(valid) / len(df),
        "mean_qed": valid["qed"].mean(),
        "mean_sa_score": valid["sa_score"].mean(),
        "mean_logp": valid["logp"].mean(),
        "lipinski_pass_rate": valid["lipinski_pass"].mean(),
    }
