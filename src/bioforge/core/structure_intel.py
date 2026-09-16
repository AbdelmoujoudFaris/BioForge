"""Intake & Structure Intelligence.

Resolves a target given as a PDB ID, UniProt accession, AlphaFold DB ID, or
raw sequence into a receptor structure, then detects candidate binding
pockets by combining a classical geometric method (a bounded LIGSITE-style
grid scan, implemented here, plus the vendored fpocket/geometric-center
fallback from `pharmaforge_core`) with a geometric-deep-learning druggability
scorer (a small, real `torch_geometric` GNN forward pass over each candidate
pocket's atom graph).

Honesty note (same policy as `engines/pharmaforge_core`): `PocketDruggabilityGNN`
runs with randomly-initialized-but-seeded weights - it demonstrates the real
architecture and a genuine forward pass, not a trained druggability predictor.
Train it against a labeled pocket-druggability dataset (e.g. PDBbind + a
negative set) before trusting its scores over the classical heuristic alone.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import requests
import torch
from sklearn.neighbors import KDTree
from torch import nn

from bioforge.common.config import get_settings
from pharmaforge_core.config import PocketConfig
from pharmaforge_core.data.pdb_download import fetch_structure
from pharmaforge_core.data.pocket_extraction import (
    Pocket,
    extract_pocket_around_ligand,
    extract_pocket_around_point,
    find_fpocket_cavities,
    geometric_center_fallback,
)
from pharmaforge_core.utils.chem import vdw_radius

logger = logging.getLogger(__name__)

TargetSourceKind = Literal["pdb_id", "uniprot_id", "alphafold_id", "sequence"]


# ---------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------


@dataclass
class ResolvedTarget:
    name: str
    pdb_path: Path
    structure_method: str
    sequence: str | None
    plddt_mean: float | None
    resolved_pdb_id: str | None
    provenance: dict


class StructurePredictorAdapter:
    """Extension point for de novo structure prediction from sequence alone.

    Not implemented in this repository: Boltz-1, Chai-1 and AlphaFold3 all
    need either a GPU-resident model (multi-GB weights) or a paid/rate-limited
    hosted API this development environment has neither of. Concrete
    subclasses should raise `NotImplementedError` with a pointer to the
    upstream project until wired to real weights/credentials - see
    `RuleBasedADMET`/`LearnedADMET` in `pharmaforge_core.scoring.admet` for the
    same honest-stub pattern.
    """

    name: str = "base"

    def predict(self, sequence: str) -> Path:
        raise NotImplementedError(
            f"{self.name} structure prediction requires GPU weights/API credentials not "
            "available in this deployment. Supply a pdb_id, uniprot_id, or alphafold_id "
            "instead, or wire this adapter to a real Boltz-1/Chai-1/AlphaFold3 backend."
        )


class Boltz1Adapter(StructurePredictorAdapter):
    name = "boltz-1"


class Chai1Adapter(StructurePredictorAdapter):
    name = "chai-1"


class AlphaFold3APIAdapter(StructurePredictorAdapter):
    name = "alphafold3-api"


def _fetch_uniprot_sequence(accession: str, timeout: float) -> str:
    settings = get_settings()
    resp = requests.get(settings.uniprot_rest_url.format(accession=accession) + ".fasta", timeout=timeout)
    resp.raise_for_status()
    lines = resp.text.strip().splitlines()
    return "".join(line.strip() for line in lines if not line.startswith(">"))


def _fetch_alphafold_structure(accession: str, cache_dir: Path, timeout: float) -> tuple[Path, float | None]:
    """Fetch an AlphaFold DB predicted structure + its mean per-residue pLDDT.

    AlphaFold DB is free/keyless (EMBL-EBI); this is a real network call, not
    a stub, when `alphafold.ebi.ac.uk` is reachable.
    """
    settings = get_settings()
    meta = requests.get(settings.alphafold_api_url.format(accession=accession), timeout=timeout)
    meta.raise_for_status()
    entries = meta.json()
    if not entries:
        raise ValueError(f"No AlphaFold DB prediction found for {accession}")
    entry = entries[0]
    pdb_url = entry["pdbUrl"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"AF-{accession}.pdb"
    if not dest.exists():
        pdb_resp = requests.get(pdb_url, timeout=timeout)
        pdb_resp.raise_for_status()
        dest.write_bytes(pdb_resp.content)

    plddt = None
    try:
        plddt_vals = [
            float(line[60:66])
            for line in dest.read_text().splitlines()
            if line.startswith("ATOM") and line[12:16].strip() == "CA"
        ]
        plddt = float(np.mean(plddt_vals)) if plddt_vals else None
    except Exception:  # pragma: no cover - defensive parsing
        plddt = None
    return dest, plddt


def resolve_target(
    source: TargetSourceKind,
    value: str,
    name: str = "target",
    timeout: float | None = None,
) -> ResolvedTarget:
    """Turn a target intake spec into a receptor structure on disk.

    * `pdb_id` - real RCSB download (ported `pharmaforge_core.data.pdb_download`).
    * `alphafold_id` - real AlphaFold DB fetch (UniProt accession -> predicted structure + pLDDT).
    * `uniprot_id` - fetches the real sequence; also tries AlphaFold DB (most UniProt entries
      have a matching AF prediction) since most downstream stages need a 3D structure.
    * `sequence` - no structure is available offline; raises with guidance rather than
      fabricating coordinates (see `StructurePredictorAdapter`).
    """
    settings = get_settings()
    timeout = timeout or settings.http_timeout_s
    cache_dir = settings.provenance_log_dir.parent / "data_cache" / "structures"

    if source == "pdb_id":
        pdb_path = fetch_structure(value)
        return ResolvedTarget(
            name=name,
            pdb_path=pdb_path,
            structure_method="experimental_pdb",
            sequence=None,
            plddt_mean=None,
            resolved_pdb_id=value.strip().upper(),
            provenance={"source": "rcsb", "pdb_id": value.strip().upper()},
        )

    if source == "alphafold_id":
        pdb_path, plddt = _fetch_alphafold_structure(value, cache_dir, timeout)
        return ResolvedTarget(
            name=name,
            pdb_path=pdb_path,
            structure_method="alphafold_db_predicted",
            sequence=None,
            plddt_mean=plddt,
            resolved_pdb_id=None,
            provenance={"source": "alphafold_db", "accession": value},
        )

    if source == "uniprot_id":
        sequence = _fetch_uniprot_sequence(value, timeout)
        try:
            pdb_path, plddt = _fetch_alphafold_structure(value, cache_dir, timeout)
            method = "alphafold_db_predicted"
        except Exception as exc:
            raise ValueError(
                f"UniProt {value}: fetched sequence ({len(sequence)} aa) but no AlphaFold DB "
                f"structure is available ({exc}). Supply a pdb_id or alphafold_id directly, or "
                "wire a StructurePredictorAdapter subclass to a real Boltz-1/Chai-1 backend."
            ) from exc
        return ResolvedTarget(
            name=name,
            pdb_path=pdb_path,
            structure_method=method,
            sequence=sequence,
            plddt_mean=plddt,
            resolved_pdb_id=None,
            provenance={"source": "uniprot+alphafold_db", "accession": value},
        )

    if source == "sequence":
        raise NotImplementedError(
            "De novo structure prediction from a raw sequence needs Boltz-1/Chai-1/AlphaFold3 "
            "(see StructurePredictorAdapter) - none are wired to real weights/credentials in this "
            "deployment. Provide a pdb_id, uniprot_id, or alphafold_id instead."
        )

    raise ValueError(f"Unknown target source: {source}")


# ---------------------------------------------------------------------------
# Classical pocket detection: bounded LIGSITE-style grid scan
# ---------------------------------------------------------------------------

_DIRECTIONS = np.array(
    [
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
        [1, 1, 1], [-1, -1, -1],
    ],
    dtype=np.float64,
)
_DIRECTIONS /= np.linalg.norm(_DIRECTIONS, axis=1, keepdims=True)
_AXIS_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]  # opposite-direction pairs for a PSP event


def ligsite_grid_pockets(
    pdb_path: Path,
    search_radius: float = 18.0,
    grid_spacing: float = 1.6,
    probe_radius: float = 1.4,
    enclosure_reach: float = 7.0,
    min_psp_axes: int = 3,
    max_candidates: int = 5,
) -> list[np.ndarray]:
    """Coarse, bounded LIGSITE-style pocket-point detector (Hendlich et al.,
    1997): grid the region within `search_radius` of the receptor's geometric
    center, mark solvent-accessible grid points, and flag a point as
    pocket-lining when protein atoms are found on *both* sides along enough
    of the four axis-pairs in `_DIRECTIONS` (a "protein-solvent-protein"
    event) - i.e. the point sits inside a concavity, not on open surface.

    Deliberately bounded to a `search_radius`-sized region (not the whole
    receptor) to keep the O(grid_points x atoms) distance computation cheap
    enough to run inline in a request handler; for a large receptor this is a
    local refinement around a seed region, not blind whole-structure cavity
    search - pair it with `find_fpocket_cavities` (real fpocket binary) for
    that when available.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    atoms = [a for a in structure.get_atoms() if a.element != "H"]
    coords = np.array([a.coord for a in atoms], dtype=np.float64)
    if len(coords) == 0:
        return [geometric_center_fallback(pdb_path)]

    center = coords.mean(axis=0)
    tree = KDTree(coords)

    n = max(4, int(2 * search_radius / grid_spacing))
    lin = np.linspace(-search_radius, search_radius, n)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1) + center

    # keep only solvent-accessible points: not inside any atom's vdw+probe radius
    dist_to_nearest, _ = tree.query(grid, k=1)
    solvent_mask = dist_to_nearest.ravel() > (1.7 + probe_radius)
    candidate_points = grid[solvent_mask]
    if len(candidate_points) == 0:
        return [center]

    psp_counts = np.zeros(len(candidate_points), dtype=int)

    # for each axis-pair, sample a few points outward along +/- direction and
    # require a protein atom within enclosure_reach on both sides
    steps = np.arange(1, int(enclosure_reach / grid_spacing) + 1) * grid_spacing
    for a_idx, b_idx in _AXIS_PAIRS:
        dir_pos = _DIRECTIONS[a_idx]
        dir_neg = _DIRECTIONS[b_idx]
        hit_pos = np.zeros(len(candidate_points), dtype=bool)
        hit_neg = np.zeros(len(candidate_points), dtype=bool)
        for step in steps:
            probe_pos = candidate_points + dir_pos * step
            probe_neg = candidate_points + dir_neg * step
            d_pos, _ = tree.query(probe_pos, k=1)
            d_neg, _ = tree.query(probe_neg, k=1)
            hit_pos |= d_pos.ravel() <= 1.9
            hit_neg |= d_neg.ravel() <= 1.9
        psp_counts += (hit_pos & hit_neg).astype(int)

    pocket_points = candidate_points[psp_counts >= min_psp_axes]
    if len(pocket_points) == 0:
        return [center]

    # cheap connected-component clustering on the grid (union by proximity)
    cluster_tree = KDTree(pocket_points)
    visited = np.zeros(len(pocket_points), dtype=bool)
    clusters: list[np.ndarray] = []
    for i in range(len(pocket_points)):
        if visited[i]:
            continue
        neighbors = cluster_tree.query_radius(pocket_points[i : i + 1], r=grid_spacing * 1.8)[0]
        stack = list(neighbors)
        member_idx = set()
        while stack:
            j = stack.pop()
            if visited[j]:
                continue
            visited[j] = True
            member_idx.add(j)
            more = cluster_tree.query_radius(pocket_points[j : j + 1], r=grid_spacing * 1.8)[0]
            stack.extend(int(m) for m in more if not visited[m])
        clusters.append(pocket_points[list(member_idx)])

    clusters.sort(key=len, reverse=True)
    return [c.mean(axis=0) for c in clusters[:max_candidates]]


# ---------------------------------------------------------------------------
# Geometric deep learning pocket-druggability scorer
# ---------------------------------------------------------------------------


class PocketDruggabilityGNN(nn.Module):
    """Small message-passing GNN over a pocket's atom graph -> druggability
    logit. Randomly initialized with a fixed seed (see `_SEEDED_MODEL`): this
    demonstrates a real geometric-deep-learning forward pass in the pocket
    detection ensemble, not a trained druggability predictor. Train against
    labeled druggable/non-druggable pockets (e.g. the fpocket-annotated
    scPDB or CryptoSite sets) before trusting it over the classical score.
    """

    def __init__(self, in_dim: int = 6, hidden_dim: int = 32):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self._global_mean_pool = global_mean_pool

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        pooled = h.mean(dim=0, keepdim=True)
        return torch.sigmoid(self.head(pooled)).squeeze()


_SEEDED_MODEL: PocketDruggabilityGNN | None = None


def _get_seeded_model() -> PocketDruggabilityGNN:
    global _SEEDED_MODEL
    if _SEEDED_MODEL is None:
        with torch.random.fork_rng():
            torch.manual_seed(1337)
            _SEEDED_MODEL = PocketDruggabilityGNN()
    return _SEEDED_MODEL


def _knn_edge_index(pos: np.ndarray, k: int) -> torch.Tensor:
    """Manual k-NN graph via `sklearn.neighbors.KDTree`, avoiding the
    `torch_geometric.nn.knn_graph` optional C++ extension (`pyg-lib` /
    `torch-cluster`), which isn't part of the base `torch_geometric` install.
    """
    n = pos.shape[0]
    k = min(k, max(1, n - 1))
    tree = KDTree(pos)
    _, neighbor_idx = tree.query(pos, k=k + 1)  # includes self at column 0
    src, dst = [], []
    for i in range(n):
        for j in neighbor_idx[i, 1:]:
            src.append(i)
            dst.append(int(j))
    return torch.tensor([src, dst], dtype=torch.long)


def _pocket_atom_graph(pocket: Pocket, k: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(pocket.elements)
    feats = np.zeros((n, 6), dtype=np.float32)
    feats[:, 0] = pocket.is_hydrophobic
    feats[:, 1] = pocket.is_hbond_donor
    feats[:, 2] = pocket.is_hbond_acceptor
    feats[:, 3] = pocket.charge
    feats[:, 4] = [vdw_radius(e) for e in pocket.elements]
    feats[:, 5] = 1.0  # bias term
    x = torch.tensor(feats)
    edge_index = _knn_edge_index(pocket.coords, k)
    return x, edge_index


def score_pocket_geometric_dl(pocket: Pocket) -> float:
    if len(pocket.elements) < 2:
        return 0.0
    model = _get_seeded_model()
    x, edge_index = _pocket_atom_graph(pocket)
    with torch.no_grad():
        return float(model(x, edge_index).item())


def score_pocket_classical(pocket: Pocket) -> float:
    """Descriptor-based druggability heuristic in [0, 1]: rewards pocket
    size and hydrophobic character (both correlate with druggability in the
    fpocket/DoGSiteScorer literature), penalizes extreme net charge (very
    polar cavities tend to bind poorly to small, mostly-neutral drug-like
    molecules).
    """
    n = len(pocket.elements)
    if n == 0:
        return 0.0
    size_term = float(np.clip(n / 150.0, 0.0, 1.0))
    hydrophobic_term = float(pocket.is_hydrophobic.mean()) if n else 0.0
    charge_penalty = float(np.clip(1.0 - abs(pocket.charge.mean()) / 2.0, 0.0, 1.0))
    return float(np.clip(0.5 * size_term + 0.35 * hydrophobic_term + 0.15 * charge_penalty, 0.0, 1.0))


@dataclass
class ScoredPocket:
    pocket: Pocket
    center: np.ndarray
    classical_score: float
    geometric_dl_score: float
    combined_score: float
    method: str


def detect_pockets(
    resolved: ResolvedTarget,
    ligand_resname: str | None = None,
    cfg: PocketConfig | None = None,
    max_candidates: int = 5,
) -> list[ScoredPocket]:
    """Ensemble pocket detection: classical (fpocket if installed, else the
    bounded LIGSITE-style grid scan above) x geometric-DL re-ranking.
    """
    cfg = cfg or PocketConfig()

    if ligand_resname:
        pocket = extract_pocket_around_ligand(resolved.pdb_path, ligand_resname, cfg)
        pockets_by_center = {tuple(pocket.center): pocket}
        method_note = "reference_ligand"
    else:
        centers = find_fpocket_cavities(resolved.pdb_path)
        method_note = "fpocket+gnn"
        if not centers:
            centers = ligsite_grid_pockets(resolved.pdb_path, max_candidates=max_candidates)
            method_note = "ligsite_grid+gnn"
        pockets_by_center = {}
        for c in centers:
            pockets_by_center[tuple(c)] = extract_pocket_around_point(resolved.pdb_path, c, cfg)

    scored = []
    for center, pocket in pockets_by_center.items():
        classical = score_pocket_classical(pocket)
        geometric = score_pocket_geometric_dl(pocket)
        combined = 0.5 * classical + 0.5 * geometric
        scored.append(
            ScoredPocket(
                pocket=pocket,
                center=np.asarray(center),
                classical_score=classical,
                geometric_dl_score=geometric,
                combined_score=combined,
                method=method_note,
            )
        )
    scored.sort(key=lambda s: s.combined_score, reverse=True)
    return scored[:max_candidates]
