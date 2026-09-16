"""Generative Chemistry microservice: ensemble generation + multi-objective ranking."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from bioforge.common.logging_config import configure_logging
from bioforge.common.provenance import track_run
from bioforge.common.schemas import Candidate, GenerationRequest, GenerationResult, ObjectiveScores
from bioforge.core.generation_ensemble import generate_ensemble

configure_logging("generate")
app = FastAPI(title="BioForge Generative Chemistry Service", version="0.1.0")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerationResult)
def generate(req: GenerationRequest):
    if req.target.source.value != "pdb_id":
        raise HTTPException(
            status_code=422,
            detail="The generation service currently requires a resolved pdb_id target "
            "(run /structure/resolve first for uniprot_id/alphafold_id targets).",
        )
    with track_run(
        "generate.ensemble", engine_versions={"pharmaforge_core": _engine_version(), "n_timesteps": str(req.n_timesteps)}
    ) as run:
        run.log_params(pdb_id=req.target.value, n_candidates=req.n_candidates, generators=[g.value for g in req.generators])
        ranked = generate_ensemble(
            pdb_id=req.target.value,
            generator_names=[g.value for g in req.generators],
            n_candidates=req.n_candidates,
            max_atoms=req.max_atoms,
            n_timesteps=req.n_timesteps,
            ligand_resname=req.target.ligand_resname,
            anti_target_pdb_ids=req.anti_targets,
            target_name=req.target.name,
            seed=req.seed,
        )
        candidates = [
            Candidate(
                candidate_id=c.candidate_id,
                smiles=c.smiles,
                valid=c.valid,
                generator_source=c.generator_source,
                scores=ObjectiveScores(
                    binding_affinity_kcal_mol=c.binding_affinity_kcal_mol,
                    sa_score=c.sa_score,
                    qed=c.qed,
                    admet_desirability=c.admet_desirability,
                ),
                pareto_optimal=c.pareto_optimal,
            )
            for c in ranked
        ]
        n_valid = sum(1 for c in candidates if c.valid)
        run.log_metrics(n_valid=n_valid, n_pareto_optimal=sum(1 for c in candidates if c.pareto_optimal))
        return GenerationResult(
            target_name=req.target.name, candidates=candidates, n_requested=req.n_candidates, n_valid=n_valid
        )


def _engine_version() -> str:
    import pharmaforge_core

    return pharmaforge_core.__version__
