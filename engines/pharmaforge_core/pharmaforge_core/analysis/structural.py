"""Structural analysis: RMSD-to-reference, interaction fingerprints,
pharmacophore matching, and PyMOL/NGLview pose-visualization script export.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

from pharmaforge_core.utils.geometry import kabsch_rmsd


def rmsd_to_reference(candidate_mol: Chem.Mol, reference_mol: Chem.Mol) -> float | None:
    """RMSD after optimal alignment; returns None if atom counts/graphs are
    too different for RDKit's `GetBestRMS` substructure-based alignment.
    """
    try:
        return rdMolAlign.GetBestRMS(candidate_mol, reference_mol)
    except (RuntimeError, ValueError):
        # Fall back to raw Kabsch RMSD over heavy-atom coordinates when the
        # molecules don't share an exact substructure match (common when
        # comparing a generated candidate against a chemically different
        # reference ligand).
        if candidate_mol.GetNumAtoms() != reference_mol.GetNumAtoms():
            return None
        p = torch.tensor(candidate_mol.GetConformer().GetPositions())
        q = torch.tensor(reference_mol.GetConformer().GetPositions())
        return kabsch_rmsd(p, q)


def pharmacophore_fingerprint(mol: Chem.Mol):
    factory = Gobbi_Pharm2D.factory
    return Generate.Gen2DFingerprint(mol, factory)


def pharmacophore_similarity(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
    from rdkit import DataStructs

    fp_a, fp_b = pharmacophore_fingerprint(mol_a), pharmacophore_fingerprint(mol_b)
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def interaction_fingerprint_overlap(fp_a: dict, fp_b: dict) -> float:
    """Cosine similarity between two {interaction_type: strength} dicts, as
    produced by `scoring.surrogate.BindingSurrogate.evaluate`.
    """
    keys = set(fp_a) | set(fp_b)
    a = np.array([fp_a.get(k, 0.0) for k in keys])
    b = np.array([fp_b.get(k, 0.0) for k in keys])
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def write_pymol_pose_script(
    pocket_pdb_path: str | Path,
    candidate_mols: list[Chem.Mol],
    output_path: str | Path,
    reference_mol: Chem.Mol | None = None,
) -> Path:
    """Emit a standalone PyMOL `.pml` script that loads the pocket and every
    candidate pose, colors them by rank, and (optionally) overlays the
    reference ligand - handed to a bench chemist or run headless
    (`pymol -cq poses.pml`) for a rendered figure.
    """
    output_path = Path(output_path)
    lines = [f"load {Path(pocket_pdb_path).as_posix()}, pocket", "hide everything, pocket", "show cartoon, pocket", "color grey80, pocket"]
    sdf_dir = output_path.parent / "candidate_sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)
    palette = ["marine", "orange", "forest", "purple", "yellow", "red", "cyan", "magenta"]

    for i, mol in enumerate(candidate_mols):
        sdf_path = sdf_dir / f"candidate_{i}.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()
        obj = f"candidate_{i}"
        color = palette[i % len(palette)]
        lines += [
            f"load {sdf_path.as_posix()}, {obj}",
            f"show sticks, {obj}",
            f"color {color}, {obj}",
        ]

    if reference_mol is not None:
        ref_path = sdf_dir / "reference.sdf"
        writer = Chem.SDWriter(str(ref_path))
        writer.write(reference_mol)
        writer.close()
        lines += [f"load {ref_path.as_posix()}, reference", "show sticks, reference", "color grey30, reference"]

    lines += ["bg_color white", "set ray_opaque_background, 0", "zoom pocket", "ray 1600, 1200"]
    output_path.write_text("\n".join(lines))
    return output_path


def write_nglview_snippet(pocket_pdb_path: str | Path, candidate_sdf_path: str | Path) -> str:
    """Returns a copy-pasteable Jupyter cell (used by
    notebooks/02_pocket_conditioned_generation.ipynb) for interactive 3D
    pose inspection with NGLview.
    """
    return (
        "import nglview as nv\n"
        "import MDAnalysis as mda\n\n"
        f"universe = mda.Universe(r'{pocket_pdb_path}')\n"
        "view = nv.show_mdanalysis(universe)\n"
        f"view.add_component(r'{candidate_sdf_path}')\n"
        "view.component_1.add_representation('licorice')\n"
        "view\n"
    )
