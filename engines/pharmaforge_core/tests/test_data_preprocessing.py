import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from pharmaforge_core.data.preprocessing import (
    PHYSCHEM_DIM,
    ligand_mol_to_tensors,
    pocket_to_tensors,
    raw_atoms_to_ligand_tensors,
)
from pharmaforge_core.data.pocket_extraction import Pocket


def test_ligand_mol_to_tensors_shapes():
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol = Chem.RemoveHs(mol)

    tensors = ligand_mol_to_tensors(mol)
    n_atoms = mol.GetNumAtoms()
    assert tensors["atom_type_idx"].shape == (n_atoms,)
    assert tensors["physchem"].shape == (n_atoms, PHYSCHEM_DIM)
    assert tensors["coords"].shape == (n_atoms, 3)


def test_ligand_mol_to_tensors_requires_conformer():
    mol = Chem.MolFromSmiles("CCO")
    try:
        ligand_mol_to_tensors(mol)
        assert False, "expected ValueError for mol without a conformer"
    except ValueError:
        pass


def test_pocket_to_tensors_shapes():
    n = 10
    pocket = Pocket(
        elements=["C"] * n,
        residue_names=["ALA"] * n,
        coords=np.random.randn(n, 3).astype(np.float32),
        is_hydrophobic=np.ones(n, dtype=bool),
        is_hbond_donor=np.zeros(n, dtype=bool),
        is_hbond_acceptor=np.zeros(n, dtype=bool),
        charge=np.zeros(n, dtype=np.float32),
        source_pdb_id="TEST",
        center=np.zeros(3, dtype=np.float32),
    )
    tensors = pocket_to_tensors(pocket)
    assert tensors["atom_type_idx"].shape == (n,)
    assert tensors["residue_type_idx"].shape == (n,)
    assert tensors["physchem"].shape == (n, PHYSCHEM_DIM)
    assert tensors["coords"].shape == (n, 3)


def test_raw_atoms_to_ligand_tensors():
    elements = ["C", "N", "O"]
    coords = np.random.randn(3, 3)
    tensors = raw_atoms_to_ligand_tensors(elements, coords)
    assert tensors["atom_type_idx"].shape == (3,)
    assert tensors["coords"].shape == (3, 3)
    assert torch.allclose(tensors["coords"], torch.as_tensor(coords, dtype=torch.float32))
