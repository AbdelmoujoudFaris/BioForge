"""Causal Mechanistic Interpretability & Systems Pharmacology microservice.

BioForge's differentiating feature - see `bioforge.core.mechanism` for the
real STRING/OmniPath/KEGG/PubMed-backed implementation.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from bioforge.common.logging_config import configure_logging
from bioforge.common.provenance import track_run
from bioforge.common.schemas import CausalChainStep, MechanisticHypothesis, PerturbationGraphSummary
from bioforge.core.mechanism import build_perturbation_graph, generate_mechanistic_hypothesis

configure_logging("mechanism")
app = FastAPI(title="BioForge Causal Mechanistic Interpretability Service", version="0.1.0")


class MechanismRequest(BaseModel):
    candidate_id: str
    compound_label: str
    primary_target: str
    off_targets: list[str] = []
    indication: str = "the modeled indication"


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/mechanism/hypothesis", response_model=MechanisticHypothesis)
def hypothesis(req: MechanismRequest):
    with track_run("mechanism.hypothesis") as run:
        run.log_params(primary_target=req.primary_target, off_targets=req.off_targets)
        result = generate_mechanistic_hypothesis(
            compound_label=req.compound_label,
            primary_target=req.primary_target,
            off_targets=req.off_targets,
            indication=req.indication,
        )
        run.log_metrics(overall_confidence=result.overall_confidence, n_causal_steps=len(result.causal_chain))
        return MechanisticHypothesis(
            candidate_id=req.candidate_id,
            target_name=req.primary_target,
            narrative=result.narrative,
            causal_chain=[
                CausalChainStep(
                    actor=s.actor,
                    relation=s.relation,
                    target=s.target,
                    confidence=s.confidence,
                    supporting_literature_count=s.supporting_literature_count,
                )
                for s in result.causal_chain
            ],
            predicted_moa=result.predicted_moa,
            resistance_mechanisms=result.resistance_mechanisms,
            biomarker_suggestions=result.biomarker_suggestions,
            overall_confidence=result.overall_confidence,
            method=result.method,
        )


@app.post("/mechanism/perturbation-graph", response_model=PerturbationGraphSummary)
def perturbation_graph(req: MechanismRequest):
    with track_run("mechanism.perturbation_graph") as run:
        pg = build_perturbation_graph(req.primary_target, req.off_targets)
        run.log_metrics(n_nodes=pg.graph.number_of_nodes(), n_edges=pg.graph.number_of_edges())
        return PerturbationGraphSummary(
            candidate_id=req.candidate_id,
            n_nodes=pg.graph.number_of_nodes(),
            n_edges=pg.graph.number_of_edges(),
            primary_target=pg.primary_target,
            off_targets=pg.off_targets,
            source=pg.source,
        )
