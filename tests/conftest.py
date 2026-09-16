from __future__ import annotations

import socket

import pytest


def _network_available(host: str = "pubchem.ncbi.nlm.nih.gov", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def network_available() -> bool:
    return _network_available()


@pytest.fixture(scope="session")
def bundled_pdb_path():
    from pharmaforge_core.config import DATA_DIR

    return DATA_DIR / "pdb" / "1CRN.pdb"
