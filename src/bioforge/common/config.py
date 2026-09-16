"""Central runtime settings, shared by every service via environment variables.

Every service reads the same `Settings` object so a docker-compose /
Kubernetes deployment only has to set env vars once (see `.env.example`)
instead of each service inventing its own config surface.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_ROOT = REPO_ROOT / "engines" / "pharmaforge_core"
DATA_CACHE = REPO_ROOT / "data_cache"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BIOFORGE_", env_file=".env", extra="ignore")

    # service discovery (docker-compose service names / k8s service DNS)
    gateway_url: str = "http://localhost:8000"
    structure_service_url: str = "http://localhost:8001"
    generate_service_url: str = "http://localhost:8002"
    refine_service_url: str = "http://localhost:8003"
    retrieval_service_url: str = "http://localhost:8004"
    admet_service_url: str = "http://localhost:8005"
    mechanism_service_url: str = "http://localhost:8006"
    report_service_url: str = "http://localhost:8007"
    orchestrator_service_url: str = "http://localhost:8008"

    # message queue / distributed compute (optional, degrades to in-process)
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"

    # provenance
    mlflow_tracking_uri: str | None = None
    provenance_log_dir: Path = REPO_ROOT / ".provenance"

    # external, free/no-key APIs used by structure + retrieval + mechanism services
    rcsb_file_url: str = "https://files.rcsb.org/download/{pdb_id}.pdb"
    uniprot_rest_url: str = "https://rest.uniprot.org/uniprotkb/{accession}"
    alphafold_api_url: str = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
    pubchem_pug_url: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    chembl_api_url: str = "https://www.ebi.ac.uk/chembl/api/data"
    string_api_url: str = "https://string-db.org/api"
    omnipath_api_url: str = "https://omnipathdb.org"

    # request-scoped
    http_timeout_s: float = 20.0

    # LLM RAG (mechanism service) - optional, offline template fallback used when unset
    llm_provider: str | None = None  # e.g. "anthropic" | "openai" | None for offline fallback
    llm_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
