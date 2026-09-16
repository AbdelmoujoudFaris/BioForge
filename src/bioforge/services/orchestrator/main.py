"""Orchestrator microservice: runs the full BioForge pipeline via the
LangGraph `StateGraph` defined in `bioforge.orchestrator.graph`.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from bioforge.common.logging_config import configure_logging
from bioforge.common.schemas import PipelineRunRequest, PipelineRunResult
from bioforge.orchestrator.graph import run_pipeline

configure_logging("orchestrator")
app = FastAPI(title="BioForge Orchestrator Service", version="0.1.0")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/pipeline/run", response_model=PipelineRunResult)
def pipeline_run(req: PipelineRunRequest):
    try:
        return run_pipeline(req)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
