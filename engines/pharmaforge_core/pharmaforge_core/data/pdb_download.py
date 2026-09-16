"""Automated PDB structure acquisition.

Thin, dependency-light wrapper around the RCSB REST/file endpoints. Kept
separate from `pocket_extraction.py` so the dashboard (ui/dashboard.py) can
call `fetch_structure("6LU7")` directly without pulling in fpocket/pdb2pqr.
"""
from __future__ import annotations

import logging
from pathlib import Path

import requests

from pharmaforge_core.config import DATA_DIR

logger = logging.getLogger(__name__)

RCSB_FILE_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
RCSB_METADATA_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"


def fetch_structure(pdb_id: str, cache_dir: Path | None = None, force: bool = False) -> Path:
    """Download a PDB structure file, caching it under `data_cache/pdb/`.

    Raises `requests.HTTPError` if the ID doesn't exist. Callers in the
    dashboard should catch that and show a friendly "PDB ID not found".
    """
    pdb_id = pdb_id.strip().upper()
    cache_dir = cache_dir or (DATA_DIR / "pdb")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{pdb_id}.pdb"
    if dest.exists() and not force:
        return dest

    url = RCSB_FILE_URL.format(pdb_id=pdb_id)
    logger.info("Downloading %s from %s", pdb_id, url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def fetch_entry_metadata(pdb_id: str) -> dict:
    pdb_id = pdb_id.strip().upper()
    response = requests.get(RCSB_METADATA_URL.format(pdb_id=pdb_id), timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_ligand_expansion_codes(pdb_id: str) -> list[str]:
    """Return the hetero-atom (ligand) 3-letter codes present in an entry,
    used to auto-pick a reference co-crystallized ligand for pocket carving.
    """
    metadata = fetch_entry_metadata(pdb_id)
    return metadata.get("rcsb_entry_container_identifiers", {}).get("non_polymer_entity_ids", [])
