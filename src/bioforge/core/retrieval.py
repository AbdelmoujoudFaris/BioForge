"""Database Retrieval & Similarity.

Hierarchical representations, in order of how "real" each is in this
repository:

  1. **ECFP4 / Tanimoto** - real, via the vendored `pharmaforge_core.utils.chem`
     Morgan-fingerprint helpers (radius=2, 2048 bits = ECFP4).
  2. **PubChem 2D similarity search** - real network call to PubChem's free,
     keyless PUG REST API (`fastsimilarity_2d`).
  3. **ChEMBL similarity search** - real network call to the ChEMBL REST API,
     with a documented graceful fallback (the public endpoint has been
     observed returning 5xx independently of this code) so one flaky
     upstream service doesn't fail the whole retrieval request.
  4. **"3D equivariant embedding"** - real forward pass through the vendored,
     E(3)-equivariant `PocketConditionedEGNN` backbone (ligand-only, no
     pocket context), used as a genuinely equivariant structural embedding.
     This stands in for Equiformer v2 / MACE, whose pretrained checkpoints
     aren't vendored here; the *architecture family* (equivariant message
     passing) is real, the weights are randomly-initialized-but-seeded, same
     honesty caveat as everywhere else untrained weights are used.
  5. **Pharmacophore matching** - real, but simplified: RDKit chemical
     feature perception (`rdMolChemicalFeatures`) reduced to a feature-count
     vector compared by cosine similarity, not a full 3D pharmacophore
     alignment engine (e.g. LigandScout).
  6. **MHFP fingerprints / contrastive embedding alignment** - documented,
     unimplemented extension points (`MHFPEncoder`, `ContrastiveAligner`):
     MHFP needs the separate `mhfp` package, and a contrastive alignment
     model needs paired multi-modal training data + a training run this
     repository doesn't ship.
  7. **DrugBank / ZINC** - DrugBank requires a commercial license/API key
     (see `DrugBankAdapter`); ZINC's public search endpoints are best-effort
     (network call attempted, falls back to a small bundled offline sample
     on failure) since this environment's access to them is unverified.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import requests
import torch
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, RDConfig

from bioforge.common.config import get_settings
from pharmaforge_core.config import EGNNConfig
from pharmaforge_core.models.egnn import PocketConditionedEGNN
from pharmaforge_core.utils.chem import morgan_fingerprint, tanimoto

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ECFP4 (real)
# ---------------------------------------------------------------------------


def ecfp4_similarity(query_smiles: str, candidate_smiles: str) -> float | None:
    query_mol = Chem.MolFromSmiles(query_smiles)
    candidate_mol = Chem.MolFromSmiles(candidate_smiles)
    if query_mol is None or candidate_mol is None:
        return None
    return float(tanimoto(query_mol, candidate_mol))


# ---------------------------------------------------------------------------
# PubChem (real)
# ---------------------------------------------------------------------------


@dataclass
class ExternalHit:
    database: str
    external_id: str
    name: str | None
    smiles: str | None
    tanimoto_ecfp4: float | None = None
    embedding_similarity: float | None = None
    pharmacophore_similarity: float | None = None
    method: str = "unknown"


def search_pubchem(query_smiles: str, threshold: int = 85, max_records: int = 10, timeout: float | None = None) -> list[ExternalHit]:
    settings = get_settings()
    timeout = timeout or settings.http_timeout_s
    base = settings.pubchem_pug_url
    try:
        cid_resp = requests.get(
            f"{base}/compound/fastsimilarity_2d/smiles/{query_smiles}/cids/JSON",
            params={"Threshold": threshold, "MaxRecords": max_records},
            timeout=timeout,
        )
        cid_resp.raise_for_status()
        cids = cid_resp.json()["IdentifierList"]["CID"][:max_records]
        if not cids:
            return []
        prop_resp = requests.get(
            f"{base}/compound/cid/{','.join(str(c) for c in cids)}/property/ConnectivitySMILES,IUPACName/JSON",
            timeout=timeout,
        )
        prop_resp.raise_for_status()
        rows = prop_resp.json()["PropertyTable"]["Properties"]
        hits = []
        for row in rows:
            smiles = row.get("ConnectivitySMILES")
            sim = ecfp4_similarity(query_smiles, smiles) if smiles else None
            hits.append(
                ExternalHit(
                    database="pubchem",
                    external_id=f"CID{row['CID']}",
                    name=row.get("IUPACName"),
                    smiles=smiles,
                    tanimoto_ecfp4=sim,
                    method="pubchem_fastsimilarity_2d",
                )
            )
        return hits
    except Exception as exc:
        logger.warning("PubChem similarity search failed: %s", exc)
        return []


def search_chembl(query_smiles: str, threshold: int = 80, max_records: int = 10, timeout: float | None = None) -> list[ExternalHit]:
    settings = get_settings()
    timeout = timeout or settings.http_timeout_s
    try:
        resp = requests.get(
            f"{settings.chembl_api_url}/similarity/{query_smiles}/{threshold}.json",
            params={"limit": max_records},
            timeout=timeout,
        )
        resp.raise_for_status()
        molecules = resp.json().get("molecules", [])
        hits = []
        for m in molecules:
            smiles = (m.get("molecule_structures") or {}).get("canonical_smiles")
            sim = ecfp4_similarity(query_smiles, smiles) if smiles else None
            hits.append(
                ExternalHit(
                    database="chembl",
                    external_id=m.get("molecule_chembl_id", "unknown"),
                    name=m.get("pref_name"),
                    smiles=smiles,
                    tanimoto_ecfp4=sim,
                    method="chembl_similarity_api",
                )
            )
        return hits
    except Exception as exc:
        logger.info("ChEMBL similarity search unavailable (%s); returning no ChEMBL hits for this call", exc)
        return []


# Small bundled offline sample so ZINC/DrugBank retrieval always returns
# *something* deterministic when the live service is unreachable/unlicensed,
# clearly labeled by `method` so callers never mistake it for a live hit.
_ZINC_SAMPLE = [
    ("ZINC000000039875", "aspirin-like", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("ZINC000001530910", "ibuprofen-like", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
    ("ZINC000000537005", "caffeine-like", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
]
_DRUGBANK_SAMPLE = [
    ("DB00945", "Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("DB01050", "Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
    ("DB00201", "Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
]


def search_zinc(query_smiles: str, max_records: int = 5, timeout: float | None = None) -> list[ExternalHit]:
    """Attempts a real network call against a public ZINC22 endpoint. This
    project has *not* verified that endpoint's exact response contract
    against live API documentation (unlike the PubChem/ChEMBL/UniProt/
    AlphaFold/STRING/OmniPath calls elsewhere in BioForge, which were checked
    against real responses) - so any failure, including an unexpected
    response shape, falls back to a tiny bundled offline sample, clearly
    labeled `method="zinc_bundled_sample_fallback"` so callers never mistake
    it for a live hit. Verify/replace the URL below against current ZINC22
    docs before relying on the live path.
    """
    try:
        resp = requests.get(
            "https://cartblanche22.docking.org/substructure",
            params={"smiles": query_smiles, "count": max_records},
            timeout=timeout or 10.0,
        )
        resp.raise_for_status()
        rows = resp.json()
        hits = [
            ExternalHit(
                database="zinc",
                external_id=r.get("zinc_id", "unknown"),
                name=None,
                smiles=r.get("smiles"),
                tanimoto_ecfp4=ecfp4_similarity(query_smiles, r.get("smiles", "")),
                method="zinc22_live_substructure",
            )
            for r in rows[:max_records]
        ]
        if hits:
            return hits
    except Exception as exc:
        logger.info("Live ZINC search unavailable (%s); using bundled sample", exc)

    return [
        ExternalHit(
            database="zinc",
            external_id=zid,
            name=name,
            smiles=smiles,
            tanimoto_ecfp4=ecfp4_similarity(query_smiles, smiles),
            method="zinc_bundled_sample_fallback",
        )
        for zid, name, smiles in _ZINC_SAMPLE[:max_records]
    ]


class DrugBankAdapter:
    """DrugBank requires a commercial data-license API key. Without
    `BIOFORGE_DRUGBANK_API_KEY` configured, returns a tiny bundled offline
    sample (clearly labeled) instead of fabricating a live-looking response.
    """

    def search(self, query_smiles: str, max_records: int = 5) -> list[ExternalHit]:
        logger.info("DrugBank API key not configured; returning bundled sample records")
        return [
            ExternalHit(
                database="drugbank",
                external_id=dbid,
                name=name,
                smiles=smiles,
                tanimoto_ecfp4=ecfp4_similarity(query_smiles, smiles),
                method="drugbank_bundled_sample_no_license",
            )
            for dbid, name, smiles in _DRUGBANK_SAMPLE[:max_records]
        ]


# ---------------------------------------------------------------------------
# "3D equivariant embedding" via the vendored EGNN backbone (real forward pass)
# ---------------------------------------------------------------------------

_EMBED_MODEL: PocketConditionedEGNN | None = None


def _get_embedding_model() -> PocketConditionedEGNN:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        with torch.random.fork_rng():
            torch.manual_seed(7)
            _EMBED_MODEL = PocketConditionedEGNN(EGNNConfig(), physchem_dim=6, time_embed=False)
        _EMBED_MODEL.eval()
    return _EMBED_MODEL


def equivariant_embedding(mol: Chem.Mol) -> np.ndarray | None:
    """Pooled node embedding from a real forward pass through the vendored
    E(3)-equivariant EGNN backbone, ligand-only (no pocket atoms - every
    atom is marked `is_ligand`). Untrained/seeded weights: see module
    docstring. Equivariance itself is verified in
    `engines/pharmaforge_core/tests/test_egnn.py`.
    """
    if mol is None or mol.GetNumConformers() == 0:
        return None
    from pharmaforge_core.data.preprocessing import ligand_mol_to_tensors

    tensors = ligand_mol_to_tensors(mol)
    n = tensors["coords"].shape[0]
    model = _get_embedding_model()
    with torch.no_grad():
        h, _ = model(
            atom_type_idx=tensors["atom_type_idx"],
            residue_type_idx=torch.zeros(n, dtype=torch.long),
            is_ligand=torch.ones(n, dtype=torch.long),
            physchem=tensors["physchem"],
            coords=tensors["coords"],
            edge_index=_full_graph_edges(n),
        )
    return h.mean(dim=0).numpy()


def _full_graph_edges(n: int) -> torch.Tensor:
    idx = torch.arange(n)
    src = idx.repeat_interleave(n)
    dst = idx.repeat(n)
    mask = src != dst
    return torch.stack([src[mask], dst[mask]])


def embedding_cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return None
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Pharmacophore matching (real, simplified feature-count comparison)
# ---------------------------------------------------------------------------

_FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(f"{RDConfig.RDDataDir}/BaseFeatures.fdef")
_PHARMACOPHORE_FAMILIES = ["Donor", "Acceptor", "Aromatic", "Hydrophobe", "PosIonizable", "NegIonizable"]


def pharmacophore_feature_vector(mol: Chem.Mol) -> np.ndarray:
    counts = dict.fromkeys(_PHARMACOPHORE_FAMILIES, 0)
    for feat in _FEATURE_FACTORY.GetFeaturesForMol(mol):
        if feat.GetFamily() in counts:
            counts[feat.GetFamily()] += 1
    return np.array([counts[f] for f in _PHARMACOPHORE_FAMILIES], dtype=np.float32)


def pharmacophore_similarity(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
    va, vb = pharmacophore_feature_vector(mol_a), pharmacophore_feature_vector(mol_b)
    return embedding_cosine_similarity(va, vb) or 0.0


# ---------------------------------------------------------------------------
# Documented extension points
# ---------------------------------------------------------------------------


class MHFPEncoder:
    """Extension point for MinHash fingerprints (Probst & Reymond, 2018).
    Not implemented: needs the separate `mhfp` package (not part of the
    core dependency set). ECFP4/Tanimoto above is the real fingerprint path.
    """

    def encode(self, mol: Chem.Mol):  # pragma: no cover
        raise NotImplementedError("pip install mhfp and wire MHFPEncoder.encode() to mhfp.encoder.MHFPEncoder")


class ContrastiveAligner:
    """Extension point for a contrastive model aligning the ECFP4/pharmacophore/
    equivariant-embedding spaces into one shared retrieval space. Not
    implemented: needs paired multi-modal training data and a training run
    this repository doesn't ship (would live under `engines/pharmaforge_core/training/`
    alongside the diffusion/scoring trainers once built).
    """

    def align(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Train a contrastive alignment model; see class docstring.")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class RetrievalReport:
    query_smiles: str
    hits: list[ExternalHit] = field(default_factory=list)


def retrieve_similar_compounds(
    query_smiles: str,
    databases: list[str] | None = None,
    max_records_per_db: int = 5,
) -> RetrievalReport:
    databases = databases or ["pubchem", "chembl", "zinc", "drugbank"]
    hits: list[ExternalHit] = []
    if "pubchem" in databases:
        hits.extend(search_pubchem(query_smiles, max_records=max_records_per_db))
    if "chembl" in databases:
        hits.extend(search_chembl(query_smiles, max_records=max_records_per_db))
    if "zinc" in databases:
        hits.extend(search_zinc(query_smiles, max_records=max_records_per_db))
    if "drugbank" in databases:
        hits.extend(DrugBankAdapter().search(query_smiles, max_records=max_records_per_db))

    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is not None:
        for hit in hits:
            if hit.smiles:
                hit_mol = Chem.MolFromSmiles(hit.smiles)
                if hit_mol is not None:
                    hit.pharmacophore_similarity = pharmacophore_similarity(query_mol, hit_mol)

    hits.sort(key=lambda h: h.tanimoto_ecfp4 or 0.0, reverse=True)
    return RetrievalReport(query_smiles=query_smiles, hits=hits)
