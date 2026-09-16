"""Database Retrieval & Similarity microservice."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from bioforge.common.logging_config import configure_logging
from bioforge.common.provenance import track_run
from bioforge.common.schemas import RetrievalHit, RetrievalResult
from bioforge.core.retrieval import retrieve_similar_compounds

configure_logging("retrieval")
app = FastAPI(title="BioForge Database Retrieval & Similarity Service", version="0.1.0")


class RetrievalRequest(BaseModel):
    candidate_id: str
    smiles: str
    databases: list[str] = ["pubchem", "chembl", "zinc", "drugbank"]
    max_records_per_db: int = 5


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/retrieval/search", response_model=RetrievalResult)
def search(req: RetrievalRequest):
    if not req.smiles:
        raise HTTPException(status_code=422, detail="smiles is required")
    with track_run("retrieval.search") as run:
        run.log_params(candidate_id=req.candidate_id, databases=req.databases)
        report = retrieve_similar_compounds(req.smiles, databases=req.databases, max_records_per_db=req.max_records_per_db)
        hits = [
            RetrievalHit(
                database=h.database,
                external_id=h.external_id,
                name=h.name,
                smiles=h.smiles,
                tanimoto_ecfp4=h.tanimoto_ecfp4,
                embedding_similarity=h.embedding_similarity,
                method=h.method,
            )
            for h in report.hits
        ]
        run.log_metrics(n_hits=len(hits))
        return RetrievalResult(candidate_id=req.candidate_id, hits=hits)
