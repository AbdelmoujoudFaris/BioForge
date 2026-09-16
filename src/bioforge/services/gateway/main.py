"""API Gateway: BioForge's single public entry point.

Exposes the full pipeline (`/pipeline/run`, delegating to the orchestrator)
plus the OpenAPI schema every SDK/client is generated against. Individual
stage services (structure/generate/refine/retrieval/admet/mechanism/report)
are reachable directly too, per `docker-compose.yml` / `k8s/`, for clients
that only need one stage.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from bioforge.common.logging_config import configure_logging
from bioforge.common.schemas import PipelineRunRequest, PipelineRunResult
from bioforge.orchestrator.graph import run_pipeline

configure_logging("gateway")
app = FastAPI(
    title="BioForge API",
    version="0.1.0",
    description="Open-source, AI-native drug repurposing and de novo design platform.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "name": "BioForge",
        "version": "0.1.0",
        "docs": "/docs",
        "services": [
            "structure", "generate", "refine", "retrieval", "admet", "mechanism", "report", "orchestrator",
        ],
    }


@app.post("/pipeline/run", response_model=PipelineRunResult)
def pipeline_run(req: PipelineRunRequest):
    try:
        return run_pipeline(req)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
