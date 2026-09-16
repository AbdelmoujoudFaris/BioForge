"""RDKit-backed cheminformatics helpers used across generation and analysis."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

_COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
}
_VDW_RADII = {
    "H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "P": 1.80,
    "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
}


def vdw_radius(element: str) -> float:
    return _VDW_RADII.get(element, 1.7)


def mol_from_coords(
    elements: Sequence[str],
    coords: np.ndarray,
    bond_perception_tolerance: float = 0.45,
) -> Chem.Mol | None:
    """Build an RDKit mol from atom types + 3D coordinates via distance-based
    bond perception (OpenBabel-style covalent-radii heuristic).

    Returns None if no chemically sane bonding graph can be formed. This is
    the bridge between the diffusion model's raw point-cloud output and every
    downstream RDKit-based metric (QED, SA, Lipinski, scaffolds, ...).
    """
    mol = Chem.RWMol()
    for elem in elements:
        atom = Chem.Atom(elem if elem in _COVALENT_RADII else "C")
        mol.AddAtom(atom)
    n = len(elements)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            r_i = _COVALENT_RADII.get(elements[i], 0.77)
            r_j = _COVALENT_RADII.get(elements[j], 0.77)
            if d <= (r_i + r_j) * (1 + bond_perception_tolerance):
                mol.AddBond(i, j, Chem.BondType.SINGLE)
    conf = Chem.Conformer(n)
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, tuple(float(v) for v in xyz))
    mol.AddConformer(conf)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol.GetMol()


def compute_physchem(mol: Chem.Mol) -> dict:
    return {
        "qed": QED.qed(mol),
        "sa_score": synthetic_accessibility_score(mol),
        "logp": Descriptors.MolLogP(mol),
        "mw": Descriptors.MolWt(mol),
        "h_bond_donors": Descriptors.NumHDonors(mol),
        "h_bond_acceptors": Descriptors.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "tpsa": Descriptors.TPSA(mol),
    }


def lipinski_pass(props: dict) -> bool:
    return (
        props["mw"] <= 500
        and props["logp"] <= 5
        and props["h_bond_donors"] <= 5
        and props["h_bond_acceptors"] <= 10
    )


def synthetic_accessibility_score(mol: Chem.Mol) -> float:
    """SAscore (Ertl & Schuffenhauer, 2009): 1 (easy) - 10 (hard).

    Falls back gracefully if the contributed RDKit `sascorer` module is not
    on the path (it ships outside the core RDKit package in `Contrib/`).
    """
    try:
        import os
        import sys

        from rdkit.Chem import RDConfig

        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer  # type: ignore

        return float(sascorer.calculateScore(mol))
    except Exception:
        # Cheap surrogate: penalize size, rings and stereocenters when the
        # real Ertl scorer isn't available in this RDKit install.
        n_atoms = mol.GetNumHeavyAtoms()
        n_rings = mol.GetRingInfo().NumRings()
        n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False))
        return float(np.clip(1.0 + 0.02 * n_atoms + 0.3 * n_rings + 0.4 * n_stereo, 1.0, 10.0))


def murcko_scaffold_smiles(mol: Chem.Mol) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold) if scaffold is not None else ""


def morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
    from rdkit import DataStructs

    fp_a, fp_b = morgan_fingerprint(mol_a), morgan_fingerprint(mol_b)
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def is_valid(mol: Chem.Mol | None) -> bool:
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False
