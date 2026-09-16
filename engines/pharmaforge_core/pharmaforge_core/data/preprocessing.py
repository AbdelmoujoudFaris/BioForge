"""Bridges raw `Pocket` / RDKit `Mol` objects into the tensor dicts every
model in `pharmaforge_core.models` expects (`atom_type_idx`, `residue_type_idx`,
`physchem`, `coords`), and defines the physicochemical feature layout used
consistently across the pocket, the ligand, and training-time datasets.
"""
from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem

from pharmaforge_core.config import ATOM_VOCAB, RESIDUE_VOCAB
from pharmaforge_core.data.pocket_extraction import Pocket
from pharmaforge_core.utils.chem import vdw_radius

# physchem feature layout shared by pocket and ligand tensors:
# [is_hydrophobic, is_hbond_donor, is_hbond_acceptor, charge, vdw_radius, is_aromatic]
PHYSCHEM_DIM = 6

_ELEMENT_TO_IDX = {sym: i for i, sym in enumerate(ATOM_VOCAB)}
_RESIDUE_TO_IDX = {name: i for i, name in enumerate(RESIDUE_VOCAB)}


def _atom_type_index(element: str) -> int:
    return _ELEMENT_TO_IDX.get(element, _ELEMENT_TO_IDX["OTHER"])


def _residue_type_index(resname: str) -> int:
    return _RESIDUE_TO_IDX.get(resname, _RESIDUE_TO_IDX["OTHER"])


def pocket_to_tensors(pocket: Pocket, device: str | torch.device = "cpu") -> dict:
    n = len(pocket.elements)
    atom_type_idx = torch.tensor([_atom_type_index(e) for e in pocket.elements], dtype=torch.long)
    residue_type_idx = torch.tensor([_residue_type_index(r) for r in pocket.residue_names], dtype=torch.long)
    physchem = torch.zeros(n, PHYSCHEM_DIM)
    physchem[:, 0] = torch.from_numpy(pocket.is_hydrophobic.astype(np.float32))
    physchem[:, 1] = torch.from_numpy(pocket.is_hbond_donor.astype(np.float32))
    physchem[:, 2] = torch.from_numpy(pocket.is_hbond_acceptor.astype(np.float32))
    physchem[:, 3] = torch.from_numpy(pocket.charge)
    physchem[:, 4] = torch.tensor([vdw_radius(e) for e in pocket.elements])
    # physchem[:, 5] (is_aromatic) left at 0 for protein residues; ring
    # aromaticity is only meaningful for the ligand side.
    coords = torch.from_numpy(pocket.coords)

    return {
        "atom_type_idx": atom_type_idx.to(device),
        "residue_type_idx": residue_type_idx.to(device),
        "physchem": physchem.to(device),
        "coords": coords.to(device),
    }


def ligand_mol_to_tensors(mol: Chem.Mol, device: str | torch.device = "cpu") -> dict:
    """Extract atom_type_idx / physchem / coords from an RDKit mol that
    already carries a 3D conformer (e.g. a co-crystallized reference ligand,
    or a candidate reconstructed from generated coordinates).
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("ligand_mol_to_tensors requires a mol with an embedded 3D conformer")
    conf = mol.GetConformer()

    atom_type_idx, physchem, coords = [], [], []

    for atom in mol.GetAtoms():
        elem = atom.GetSymbol()
        atom_type_idx.append(_atom_type_index(elem))
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
        is_donor = elem in ("N", "O") and atom.GetTotalNumHs() > 0
        is_acceptor = elem in ("N", "O") and atom.GetTotalNumHs() == 0
        row = [
            float(elem == "C" and not atom.GetIsAromatic()),  # crude hydrophobic proxy
            float(is_donor),
            float(is_acceptor),
            float(atom.GetFormalCharge()),
            vdw_radius(elem),
            float(atom.GetIsAromatic()),
        ]
        physchem.append(row)

    return {
        "atom_type_idx": torch.tensor(atom_type_idx, dtype=torch.long, device=device),
        "physchem": torch.tensor(physchem, dtype=torch.float32, device=device),
        "coords": torch.tensor(coords, dtype=torch.float32, device=device),
    }


def raw_atoms_to_ligand_tensors(
    elements: list[str], coords: np.ndarray, device: str | torch.device = "cpu"
) -> dict:
    """Minimal path used right after diffusion sampling, before/instead of
    RDKit bond perception - lets the scoring stack evaluate a raw point
    cloud immediately without needing a valid RDKit mol first.
    """
    atom_type_idx = torch.tensor([_atom_type_index(e) for e in elements], dtype=torch.long, device=device)
    physchem = torch.zeros(len(elements), PHYSCHEM_DIM, device=device)
    physchem[:, 4] = torch.tensor([vdw_radius(e) for e in elements], device=device)
    return {
        "atom_type_idx": atom_type_idx,
        "physchem": physchem,
        "coords": torch.as_tensor(coords, dtype=torch.float32, device=device),
    }
