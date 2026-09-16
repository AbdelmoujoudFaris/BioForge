import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from bioforge.core.refinement import equilibrate_ligand, mc_dropout_uncertainty, refine_pose
from pharmaforge_core.config import ScoringConfig
from pharmaforge_core.models.scoring import ScoringStack


def _embed(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)
    return Chem.RemoveHs(mol)


def test_equilibrate_ligand_lowers_or_maintains_energy():
    mol = _embed("c1ccccc1CCN")
    result = equilibrate_ligand(mol, n_md_steps=50)
    assert result.n_atoms == mol.GetNumAtoms()
    assert result.final_potential_energy_kj_mol <= result.initial_potential_energy_kj_mol + 1e-3
    assert result.converged is True


def test_refine_pose_and_uncertainty_shapes():
    stack = ScoringStack(ScoringConfig())
    n = 6
    pocket = {
        "atom_type_idx": torch.randint(1, 9, (10,)),
        "residue_type_idx": torch.randint(1, 5, (10,)),
        "physchem": torch.rand(10, 6),
        "coords": torch.randn(10, 3) * 5,
    }
    lig_atom_type_idx = torch.randint(1, 9, (n,))
    lig_physchem = torch.rand(n, 6)
    lig_coords = torch.randn(n, 3)

    pose = refine_pose(stack, pocket, lig_atom_type_idx, lig_physchem, lig_coords, n_steps=5)
    assert pose.coords.shape == (n, 3)

    uncertainty = mc_dropout_uncertainty(
        stack, pocket, lig_atom_type_idx, lig_physchem, torch.as_tensor(pose.coords, dtype=torch.float32), n_samples=5
    )
    assert uncertainty.n_samples == 5
    assert uncertainty.std >= 0.0
