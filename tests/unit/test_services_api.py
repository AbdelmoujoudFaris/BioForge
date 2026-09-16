import importlib

import pytest
from fastapi.testclient import TestClient

SERVICE_MODULES = [
    "bioforge.services.gateway.main",
    "bioforge.services.structure.main",
    "bioforge.services.generate.main",
    "bioforge.services.refine.main",
    "bioforge.services.retrieval.main",
    "bioforge.services.admet.main",
    "bioforge.services.mechanism.main",
    "bioforge.services.report.main",
    "bioforge.services.orchestrator.main",
]


@pytest.mark.parametrize("module_path", SERVICE_MODULES)
def test_healthz(module_path):
    module = importlib.import_module(module_path)
    client = TestClient(module.app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_admet_profile_endpoint_without_off_target_panel():
    module = importlib.import_module("bioforge.services.admet.main")
    client = TestClient(module.app)
    resp = client.post(
        "/admet/profile",
        json={"candidate_id": "c1", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "run_off_target_panel": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["candidate_id"] == "c1"
    assert body["method"] == "rule_based_v1"


def test_admet_profile_endpoint_rejects_invalid_smiles():
    module = importlib.import_module("bioforge.services.admet.main")
    client = TestClient(module.app)
    resp = client.post("/admet/profile", json={"candidate_id": "c1", "smiles": "not a smiles", "run_off_target_panel": False})
    assert resp.status_code == 422


def test_report_synthesis_and_clinical_score_endpoints():
    module = importlib.import_module("bioforge.services.report.main")
    client = TestClient(module.app)

    resp = client.post("/report/synthesis", json={"candidate_id": "c1", "smiles": "CCO"})
    assert resp.status_code == 200
    assert resp.json()["sa_score"] > 0

    resp = client.post(
        "/report/clinical-translation-score",
        json={
            "candidate_id": "c1", "mechanism_confidence": 0.7, "has_known_human_exposure_analog": True,
            "admet_favorable": True, "sa_score": 3.0,
        },
    )
    assert resp.status_code == 200
    assert 0.0 <= resp.json()["composite_score"] <= 1.0
