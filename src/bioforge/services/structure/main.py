"""Structure Intelligence microservice: target intake + pocket detection."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from bioforge.common.logging_config import configure_logging
from bioforge.common.provenance import track_run
from bioforge.common.schemas import PocketCandidate, PocketResult, StructureRecord, TargetIntake
from bioforge.core.structure_intel import detect_pockets, resolve_target

configure_logging("structure")
app = FastAPI(title="BioForge Structure Intelligence Service", version="0.1.0")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/structure/resolve", response_model=StructureRecord)
def resolve(target: TargetIntake):
    with track_run("structure.resolve") as run:
        run.log_params(source=target.source, value=target.value)
        try:
            resolved = resolve_target(target.source.value, target.value, name=target.name)
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        record = StructureRecord(
            target_name=resolved.name,
            resolved_pdb_id=resolved.resolved_pdb_id,
            structure_method=resolved.structure_method,
            sequence=resolved.sequence,
            pdb_path=str(resolved.pdb_path),
            plddt_mean=resolved.plddt_mean,
            provenance=resolved.provenance,
        )
        run.log_artifact_ref("pdb_path", record.pdb_path)
        return record


@app.post("/structure/pockets", response_model=PocketResult)
def pockets(target: TargetIntake, max_candidates: int = 5):
    with track_run("structure.pockets") as run:
        try:
            resolved = resolve_target(target.source.value, target.value, name=target.name)
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        scored = detect_pockets(resolved, ligand_resname=target.ligand_resname, max_candidates=max_candidates)
        if not scored:
            raise HTTPException(status_code=422, detail="No pockets detected")

        candidates = [
            PocketCandidate(
                pocket_id=f"pocket_{i}",
                center=tuple(float(v) for v in s.center),
                n_atoms=len(s.pocket.elements),
                classical_score=s.classical_score,
                geometric_dl_score=s.geometric_dl_score,
                combined_score=s.combined_score,
                method=s.method,
            )
            for i, s in enumerate(scored)
        ]
        run.log_metrics(n_candidates=len(candidates), top_combined_score=candidates[0].combined_score)
        return PocketResult(target_name=target.name, candidates=candidates, selected_pocket_id="pocket_0")
