"""Fragment-growing / scaffold-hopping generation mode.

Starting point is a seed molecule (or a single fragment/scaffold), embedded
in the target pocket. Growth proceeds with `DifferentiableTreeSearch`
(models/actor_critic.py): at each step the actor proposes candidate next
atoms, new positions are placed using standard bond-length/tetrahedral-angle
priors relative to the growth point (rather than the naive random offset
used in the actor-critic module's own docstring example), and the critic
(scoring stack) ranks the resulting partial molecules directly - no docking,
no rollout.

Scaffold hopping is the same growth procedure seeded from a Murcko scaffold
extracted from a reference ligand instead of the full molecule, letting the
search explore alternative substituents/ring systems around a fixed core.
"""
from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from pharmaforge_core.config import ATOM_VOCAB, GenerationConfig
from pharmaforge_core.data.preprocessing import ligand_mol_to_tensors
from pharmaforge_core.models.actor_critic import ActorCritic, DifferentiableTreeSearch
from pharmaforge_core.utils.chem import is_valid, mol_from_coords

_TYPICAL_BOND_LENGTH = {  # angstrom, heavy-atom single bonds, generic default 1.5
    "C": 1.54, "N": 1.47, "O": 1.43, "S": 1.81, "F": 1.35, "Cl": 1.77, "Br": 1.94,
}
_TETRAHEDRAL_ANGLE = np.deg2rad(109.5)


def _place_new_atom(existing_coords: np.ndarray, attach_idx: int, new_element: str, rng: np.random.Generator) -> np.ndarray:
    """Places a new atom at a chemically plausible bond length/direction
    from the attachment point, picking a direction that avoids existing
    atoms (steric-aware placement, not just "extend from centroid").
    """
    attach_point = existing_coords[attach_idx]
    bond_length = _TYPICAL_BOND_LENGTH.get(new_element, 1.5)

    best_dir, best_min_dist = None, -np.inf
    for _ in range(16):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction) + 1e-8
        candidate = attach_point + direction * bond_length
        min_dist = np.min(np.linalg.norm(existing_coords - candidate, axis=1))
        if min_dist > best_min_dist:
            best_min_dist, best_dir = min_dist, direction
    return attach_point + best_dir * bond_length


def seed_from_smiles(smiles: str, n_conformers: int = 1, seed: int = 0):
    """Embed a seed molecule (or fragment) in 3D, returning its RDKit mol
    with a conformer ready for `ligand_mol_to_tensors`.
    """
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse seed SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    return Chem.RemoveHs(mol)


def scaffold_from_reference(reference_mol: Chem.Mol):
    scaffold = MurckoScaffold.GetScaffoldForMol(reference_mol)
    return scaffold


class FragmentGrower:
    def __init__(self, actor_critic: ActorCritic, beam_width: int = 8, max_depth: int = 40, seed: int = 0):
        self.rng = np.random.default_rng(seed)

        def placement_fn(coords: torch.Tensor, attach_idx: int, new_atom_type: int) -> torch.Tensor:
            element = ATOM_VOCAB[new_atom_type] if 0 <= new_atom_type < len(ATOM_VOCAB) else "C"
            new_coord = _place_new_atom(coords.numpy(), attach_idx, element, self.rng)
            return torch.as_tensor(new_coord, dtype=coords.dtype)

        self.search = DifferentiableTreeSearch(
            actor_critic.actor,
            actor_critic.critic,
            beam_width=beam_width,
            max_depth=max_depth,
            placement_fn=placement_fn,
        )

    def grow(self, pocket: dict, seed_mol: Chem.Mol, cfg: GenerationConfig) -> list[dict]:
        seed_tensors = ligand_mol_to_tensors(seed_mol)
        beam = self.search.search(
            pocket, seed_tensors["atom_type_idx"], seed_tensors["physchem"], seed_tensors["coords"]
        )
        results = []
        for atom_type_idx, physchem, coords, score in beam:
            elements = [ATOM_VOCAB[i] if 0 <= i < len(ATOM_VOCAB) else "C" for i in atom_type_idx.tolist()]
            mol = mol_from_coords(elements, coords.numpy())
            results.append({"elements": elements, "coords": coords, "score": score, "mol": mol, "valid": is_valid(mol)})
        return results
