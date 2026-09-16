import torch
import pytest

from pharmaforge_core.config import ATOM_VOCAB, RESIDUE_VOCAB, EGNNConfig


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)


@pytest.fixture
def small_egnn_config():
    return EGNNConfig(node_feature_dim=16, hidden_dim=16, n_layers=2, edge_feature_dim=0)


@pytest.fixture
def toy_pocket():
    n_pocket = 12
    return {
        "atom_type_idx": torch.randint(1, len(ATOM_VOCAB) - 1, (n_pocket,)),
        "residue_type_idx": torch.randint(1, len(RESIDUE_VOCAB) - 1, (n_pocket,)),
        "physchem": torch.randn(n_pocket, 6),
        "coords": torch.randn(n_pocket, 3) * 6.0,
    }


@pytest.fixture
def toy_ligand():
    n_ligand = 6
    return {
        "atom_type_idx": torch.randint(1, len(ATOM_VOCAB) - 1, (n_ligand,)),
        "physchem": torch.randn(n_ligand, 6),
        "coords": torch.randn(n_ligand, 3) * 2.0,
    }
