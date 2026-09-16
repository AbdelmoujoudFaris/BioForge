"""Physics-Based Refinement microservice: OpenMM equilibration, gradient pose
refinement (GNINA/DiffDock substitute), and MC-dropout pose uncertainty.
"""
from __future__ import annotations

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import AllChem

from bioforge.common.logging_config import configure_logging
from bioforge.common.provenance import track_run
from bioforge.common.schemas import PoseUncertainty, RefinementResult
from bioforge.core.refinement import equilibrate_ligand, mc_dropout_uncertainty, refine_pose
from pharmaforge_core.config import PocketConfig, ScoringConfig
from pharmaforge_core.data.pdb_download import fetch_structure
from pharmaforge_core.data.pocket_extraction import (
    extract_pocket_around_ligand,
    extract_pocket_around_point,
    geometric_center_fallback,
)
from pharmaforge_core.data.preprocessing import ligand_mol_to_tensors, pocket_to_tensors
from pharmaforge_core.models.scoring import ScoringStack

configure_logging("refine")
app = FastAPI(title="BioForge Physics-Based Refinement Service", version="0.1.0")

_scoring_stack = ScoringStack(ScoringConfig())


class RefineRequest(BaseModel):
    candidate_id: str
    smiles: str
    pdb_id: str
    ligand_resname: str | None = None
    n_md_steps: int = 200
    n_pose_refinement_steps: int = 30
    n_uncertainty_samples: int = 20


def _embed_3d(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        raise HTTPException(status_code=422, detail="RDKit 3D embedding failed for this molecule")
    AllChem.MMFFOptimizeMolecule(mol)
    return Chem.RemoveHs(mol)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/refine", response_model=RefinementResult)
def refine(req: RefineRequest):
    with track_run("refine.pipeline") as run:
        run.log_params(candidate_id=req.candidate_id, pdb_id=req.pdb_id)
        mol = _embed_3d(req.smiles)

        equilibration = equilibrate_ligand(mol, n_md_steps=req.n_md_steps)

        try:
            pdb_path = fetch_structure(req.pdb_id)
            if req.ligand_resname:
                pocket = extract_pocket_around_ligand(pdb_path, req.ligand_resname, PocketConfig())
            else:
                center = geometric_center_fallback(pdb_path)
                pocket = extract_pocket_around_point(pdb_path, center, PocketConfig())
            pocket_tensors = pocket_to_tensors(pocket)
            ligand_tensors = ligand_mol_to_tensors(mol)

            pose = refine_pose(
                _scoring_stack,
                pocket_tensors,
                ligand_tensors["atom_type_idx"],
                ligand_tensors["physchem"],
                ligand_tensors["coords"],
                n_steps=req.n_pose_refinement_steps,
            )
            uncertainty = mc_dropout_uncertainty(
                _scoring_stack,
                pocket_tensors,
                ligand_tensors["atom_type_idx"],
                ligand_tensors["physchem"],
                torch.as_tensor(pose.coords, dtype=torch.float32),
                n_samples=req.n_uncertainty_samples,
            )
            docking_score = pose.final_score
            pose_uncertainty = PoseUncertainty(
                method=uncertainty.method,
                mean=uncertainty.mean,
                std=uncertainty.std,
                n_samples=uncertainty.n_samples,
                flagged_low_confidence=uncertainty.flagged_low_confidence,
            )
        except Exception as exc:
            run.log_artifact_ref("pose_refinement_error", str(exc))
            docking_score, pose_uncertainty = None, None

        run.log_metrics(
            rmsd_to_input_angstrom=equilibration.rmsd_to_input_angstrom,
            docking_score=docking_score or 0.0,
        )
        return RefinementResult(
            candidate_id=req.candidate_id,
            docking_score_kcal_mol=docking_score,
            pose_uncertainty=pose_uncertainty,
            md_equilibration={
                "method": equilibration.method,
                "n_atoms": equilibration.n_atoms,
                "initial_potential_energy_kj_mol": equilibration.initial_potential_energy_kj_mol,
                "final_potential_energy_kj_mol": equilibration.final_potential_energy_kj_mol,
                "rmsd_to_input_angstrom": equilibration.rmsd_to_input_angstrom,
                "converged": equilibration.converged,
            },
            fep_estimate=None,
            method="openmm_generic_ff+differentiable_scoring_gradient_refinement",
        )
