"""ADMET & Safety microservice."""
from __future__ import annotations

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rdkit import Chem

from bioforge.common.logging_config import configure_logging
from bioforge.common.provenance import track_run
from bioforge.common.schemas import ADMETProfile
from bioforge.core.admet_safety import DEFAULT_ANTI_TARGET_PANEL, off_target_panel_screen, profile_admet
from pharmaforge_core.config import ScoringConfig
from pharmaforge_core.data.preprocessing import ligand_mol_to_tensors
from pharmaforge_core.models.scoring import ScoringStack
from rdkit.Chem import AllChem

configure_logging("admet")
app = FastAPI(title="BioForge ADMET & Safety Service", version="0.1.0")

_scoring_stack = ScoringStack(ScoringConfig())


class ADMETRequest(BaseModel):
    candidate_id: str
    smiles: str
    run_off_target_panel: bool = True
    anti_target_panel: dict[str, str] | None = None


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/admet/profile", response_model=ADMETProfile)
def profile(req: ADMETRequest):
    mol = Chem.MolFromSmiles(req.smiles)
    if mol is None:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: {req.smiles!r}")

    with track_run("admet.profile") as run:
        run.log_params(candidate_id=req.candidate_id)
        result = profile_admet(mol)

        off_target_hits: list[dict] = []
        if req.run_off_target_panel:
            try:
                mol_3d = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol_3d)
                mol_3d = Chem.RemoveHs(mol_3d)
                tensors = ligand_mol_to_tensors(mol_3d)
                off_target_hits = off_target_panel_screen(
                    _scoring_stack,
                    tensors["atom_type_idx"],
                    tensors["physchem"],
                    tensors["coords"],
                    panel_pdb_ids=req.anti_target_panel or DEFAULT_ANTI_TARGET_PANEL,
                )
            except Exception as exc:
                run.log_artifact_ref("off_target_panel_error", str(exc))

        run.log_metrics(n_off_target_flags=sum(1 for h in off_target_hits if h.get("flagged")))
        return ADMETProfile(
            candidate_id=req.candidate_id,
            human_intestinal_absorption=result.human_intestinal_absorption,
            blood_brain_barrier_permeant=result.blood_brain_barrier_permeant,
            cyp_inhibition_risk=result.cyp_inhibition_risk,
            herg_liability_risk=result.herg_liability_risk,
            off_target_hits=off_target_hits,
            method=result.method,
        )
