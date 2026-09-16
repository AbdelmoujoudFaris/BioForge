import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from bioforge.core.retrieval import (
    ecfp4_similarity,
    embedding_cosine_similarity,
    equivariant_embedding,
    pharmacophore_similarity,
    search_zinc,
)


def test_ecfp4_similarity_identical_molecules_is_one():
    assert ecfp4_similarity("CCO", "CCO") == pytest.approx(1.0)


def test_ecfp4_similarity_invalid_smiles_returns_none():
    assert ecfp4_similarity("not a smiles", "CCO") is None


def test_pharmacophore_similarity_self_is_one():
    mol = Chem.MolFromSmiles("c1ccccc1O")
    assert pharmacophore_similarity(mol, mol) == pytest.approx(1.0)


def _embed3d(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    return Chem.RemoveHs(mol)


def test_equivariant_embedding_is_reproducible():
    mol = _embed3d("CCO")
    e1 = equivariant_embedding(mol)
    e2 = equivariant_embedding(mol)
    assert embedding_cosine_similarity(e1, e2) == pytest.approx(1.0, abs=1e-4)


def test_search_zinc_falls_back_to_bundled_sample_when_live_call_fails(monkeypatch):
    import requests

    def _boom(*args, **kwargs):
        raise requests.ConnectionError("simulated offline")

    monkeypatch.setattr(requests, "get", _boom)
    hits = search_zinc("CCO", max_records=2)
    assert len(hits) == 2
    assert all(h.method == "zinc_bundled_sample_fallback" for h in hits)
