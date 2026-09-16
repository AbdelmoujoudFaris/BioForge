"""Advanced Analysis & Reporting microservice: synthetic tractability, patent
landscape, Clinical Translation Score, and polypharmacology graph data.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rdkit import Chem

from bioforge.common.logging_config import configure_logging
from bioforge.common.provenance import track_run
from bioforge.common.schemas import ClinicalTranslationScore, PatentLandscapeNote, SynthesisRoute
from bioforge.core.reporting import (
    clinical_translation_score,
    patent_landscape_check,
    polypharmacology_graph_data,
    synthesis_route,
)

configure_logging("report")
app = FastAPI(title="BioForge Advanced Analysis & Reporting Service", version="0.1.0")


class SynthesisRequest(BaseModel):
    candidate_id: str
    smiles: str


class PatentRequest(BaseModel):
    candidate_id: str
    smiles: str


class ClinicalScoreRequest(BaseModel):
    candidate_id: str
    mechanism_confidence: float
    has_known_human_exposure_analog: bool
    admet_favorable: bool
    sa_score: float


class PolypharmacologyRequest(BaseModel):
    compound_label: str
    primary_target: str
    off_target_scores: list[dict] = []


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/report/synthesis", response_model=SynthesisRoute)
def synthesis(req: SynthesisRequest):
    mol = Chem.MolFromSmiles(req.smiles)
    if mol is None:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: {req.smiles!r}")
    with track_run("report.synthesis"):
        result = synthesis_route(mol)
        return SynthesisRoute(
            method=result["method"], route_found=result["route_found"], n_steps=result["n_steps"], sa_score=result["sa_score"]
        )


@app.post("/report/patent", response_model=PatentLandscapeNote)
def patent(req: PatentRequest):
    with track_run("report.patent"):
        note = patent_landscape_check(req.smiles)
        return PatentLandscapeNote(method=note.method, likely_novel=note.likely_novel, notes=note.notes)


@app.post("/report/clinical-translation-score", response_model=ClinicalTranslationScore)
def clinical_score(req: ClinicalScoreRequest):
    with track_run("report.clinical_translation_score"):
        result = clinical_translation_score(
            mechanism_confidence=req.mechanism_confidence,
            has_known_human_exposure_analog=req.has_known_human_exposure_analog,
            admet_favorable=req.admet_favorable,
            sa_score=req.sa_score,
        )
        return ClinicalTranslationScore(
            candidate_id=req.candidate_id, composite_score=result.composite_score, components=result.components, method=result.method
        )


@app.post("/report/polypharmacology-graph")
def polypharmacology_graph(req: PolypharmacologyRequest):
    with track_run("report.polypharmacology_graph"):
        data = polypharmacology_graph_data(req.compound_label, req.primary_target, req.off_target_scores)
        return {"nodes": data.nodes, "edges": data.edges}
