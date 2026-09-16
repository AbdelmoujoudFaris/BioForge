"""Diversity & novelty metrics: Bemis-Murcko scaffold diversity, Tanimoto
similarity distributions, molecular clustering, and novelty against a
reference set (ChEMBL/ZINC subsample or the training set itself).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.ML.Cluster import Butina

from pharmaforge_core.utils.chem import morgan_fingerprint, murcko_scaffold_smiles


def scaffold_diversity(mols: list[Chem.Mol]) -> dict:
    scaffolds = [murcko_scaffold_smiles(m) for m in mols if m is not None]
    counts = Counter(scaffolds)
    n = len(scaffolds)
    return {
        "n_molecules": n,
        "n_unique_scaffolds": len(counts),
        "scaffold_diversity_ratio": len(counts) / n if n else 0.0,
        "top_scaffolds": counts.most_common(10),
    }


def pairwise_tanimoto_matrix(mols: list[Chem.Mol]) -> np.ndarray:
    fps = [morgan_fingerprint(m) for m in mols if m is not None]
    n = len(fps)
    mat = np.eye(n)
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        for j, s in enumerate(sims, start=i + 1):
            mat[i, j] = mat[j, i] = s
    return mat


def internal_diversity(mols: list[Chem.Mol]) -> float:
    """1 - mean pairwise Tanimoto similarity; higher = more diverse set."""
    mat = pairwise_tanimoto_matrix(mols)
    n = mat.shape[0]
    if n < 2:
        return 0.0
    off_diag = mat[np.triu_indices(n, k=1)]
    return float(1.0 - off_diag.mean())


def butina_clusters(mols: list[Chem.Mol], distance_threshold: float = 0.35) -> list[list[int]]:
    fps = [morgan_fingerprint(m) for m in mols if m is not None]
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1 - s for s in sims)
    clusters = Butina.ClusterData(dists, n, distance_threshold, isDistData=True)
    return [list(c) for c in clusters]


def novelty_against_reference(mols: list[Chem.Mol], reference_smiles: list[str], similarity_threshold: float = 0.4) -> dict:
    """Fraction of candidates with no reference-set molecule above
    `similarity_threshold` Tanimoto similarity - the standard "novelty"
    metric reported alongside validity/uniqueness/diversity in de novo
    generation benchmarks (Polykovskiy et al., MOSES; Preuer et al., FCD).
    """
    ref_mols = [Chem.MolFromSmiles(s) for s in reference_smiles]
    ref_fps = [morgan_fingerprint(m) for m in ref_mols if m is not None]
    if not ref_fps:
        return {"novelty_rate": None, "note": "empty/invalid reference set"}

    novel_flags = []
    for mol in mols:
        if mol is None:
            continue
        fp = morgan_fingerprint(mol)
        max_sim = max(DataStructs.BulkTanimotoSimilarity(fp, ref_fps))
        novel_flags.append(max_sim < similarity_threshold)
    return {
        "novelty_rate": float(np.mean(novel_flags)) if novel_flags else None,
        "n_evaluated": len(novel_flags),
        "similarity_threshold": similarity_threshold,
    }


def uniqueness(mols: list[Chem.Mol]) -> float:
    smiles = [Chem.MolToSmiles(m) for m in mols if m is not None]
    return len(set(smiles)) / len(smiles) if smiles else 0.0
