"""Python SDK for the BioForge REST API.

```python
from bioforge.sdk import BioForgeClient
from bioforge.common.schemas import TargetIntake, TargetSource, PipelineRunRequest

client = BioForgeClient(base_url="http://localhost:8000")
result = client.run_pipeline(
    PipelineRunRequest(target=TargetIntake(source=TargetSource.PDB_ID, value="6LU7", name="Mpro"))
)
```
"""
from __future__ import annotations

import httpx

from bioforge.common.schemas import (
    ADMETProfile,
    GenerationRequest,
    GenerationResult,
    PipelineRunRequest,
    PipelineRunResult,
    PocketResult,
    RefinementResult,
    RetrievalResult,
    StructureRecord,
    TargetIntake,
)


class BioForgeClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 120.0):
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BioForgeClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- gateway / orchestrator --------------------------------------------------

    def run_pipeline(self, req: PipelineRunRequest) -> PipelineRunResult:
        resp = self._client.post("/pipeline/run", json=req.model_dump(mode="json"))
        resp.raise_for_status()
        return PipelineRunResult.model_validate(resp.json())

    # -- individual stage calls (point these at each service's own base_url
    #    in a distributed deployment; against the gateway they 404 unless the
    #    gateway also proxies them - see services/gateway/main.py) ------------

    def resolve_structure(self, target: TargetIntake) -> StructureRecord:
        resp = self._client.post("/structure/resolve", json=target.model_dump(mode="json"))
        resp.raise_for_status()
        return StructureRecord.model_validate(resp.json())

    def detect_pockets(self, target: TargetIntake) -> PocketResult:
        resp = self._client.post("/structure/pockets", json=target.model_dump(mode="json"))
        resp.raise_for_status()
        return PocketResult.model_validate(resp.json())

    def generate(self, req: GenerationRequest) -> GenerationResult:
        resp = self._client.post("/generate", json=req.model_dump(mode="json"))
        resp.raise_for_status()
        return GenerationResult.model_validate(resp.json())

    def refine(self, candidate_id: str, smiles: str, pdb_id: str, **kwargs) -> RefinementResult:
        payload = {"candidate_id": candidate_id, "smiles": smiles, "pdb_id": pdb_id, **kwargs}
        resp = self._client.post("/refine", json=payload)
        resp.raise_for_status()
        return RefinementResult.model_validate(resp.json())

    def retrieve(self, candidate_id: str, smiles: str, **kwargs) -> RetrievalResult:
        payload = {"candidate_id": candidate_id, "smiles": smiles, **kwargs}
        resp = self._client.post("/retrieval/search", json=payload)
        resp.raise_for_status()
        return RetrievalResult.model_validate(resp.json())

    def admet_profile(self, candidate_id: str, smiles: str, **kwargs) -> ADMETProfile:
        payload = {"candidate_id": candidate_id, "smiles": smiles, **kwargs}
        resp = self._client.post("/admet/profile", json=payload)
        resp.raise_for_status()
        return ADMETProfile.model_validate(resp.json())
