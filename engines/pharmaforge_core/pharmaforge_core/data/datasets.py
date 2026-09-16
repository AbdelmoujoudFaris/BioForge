"""PyTorch `Dataset` wrappers for PDBbind and CrossDocked2020-style
complex archives, and a small synthetic dataset for tests/CI where the
real archives (tens of GB) aren't downloaded.

Directory layout expected for each real dataset (both distribute this
shape after their standard extraction scripts):

    <root>/<complex_id>/<complex_id>_protein.pdb
    <root>/<complex_id>/<complex_id>_ligand.sdf

`PDBBindDataset` and `CrossDocked2020Dataset` differ only in the file
naming convention and default pocket radius, so both subclass
`_ComplexDataset` and just override `_locate_files`.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from rdkit import Chem
from torch.utils.data import Dataset

from pharmaforge_core.config import PocketConfig
from pharmaforge_core.data.preprocessing import ligand_mol_to_tensors, pocket_to_tensors


class _ComplexDataset(Dataset):
    def __init__(self, root: str | Path, pocket_cfg: PocketConfig | None = None, transform: Callable | None = None):
        self.root = Path(root)
        self.pocket_cfg = pocket_cfg or PocketConfig()
        self.transform = transform
        self.complex_ids = sorted(p.name for p in self.root.iterdir() if p.is_dir()) if self.root.exists() else []

    def __len__(self) -> int:
        return len(self.complex_ids)

    def _locate_files(self, complex_id: str) -> tuple[Path, Path]:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        complex_id = self.complex_ids[idx]
        protein_path, ligand_path = self._locate_files(complex_id)

        supplier = Chem.SDMolSupplier(str(ligand_path), removeHs=False)
        ligand_mol = next((m for m in supplier if m is not None), None)
        if ligand_mol is None:
            raise ValueError(f"Could not parse ligand for complex {complex_id}")

        # Reference ligand's own resname isn't reliably in the PDB when the
        # ligand is supplied as a separate SDF; carve the pocket by proximity
        # to the ligand's 3D coordinates directly instead of by resname.
        conf = ligand_mol.GetConformer()
        ligand_coords = conf.GetPositions()
        center = ligand_coords.mean(axis=0)

        from pharmaforge_core.data.pocket_extraction import extract_pocket_around_point

        pocket = extract_pocket_around_point(protein_path, center, self.pocket_cfg)

        sample = {
            "complex_id": complex_id,
            "pocket": pocket_to_tensors(pocket),
            "ligand": ligand_mol_to_tensors(ligand_mol),
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


class PDBBindDataset(_ComplexDataset):
    def _locate_files(self, complex_id: str) -> tuple[Path, Path]:
        d = self.root / complex_id
        return d / f"{complex_id}_protein.pdb", d / f"{complex_id}_ligand.sdf"


class CrossDocked2020Dataset(_ComplexDataset):
    def _locate_files(self, complex_id: str) -> tuple[Path, Path]:
        d = self.root / complex_id
        return d / f"{complex_id}_rec.pdb", d / f"{complex_id}_lig.sdf"


def collate_single(batch: list[dict]) -> dict:
    """PharmaForge's models process one pocket+ligand graph per forward call
    (batching happens at the edge-index/graph level, not via zero-padding),
    so the default collate is simply "batch size 1, unwrap the list" - this
    keeps `DataLoader(..., batch_size=1, collate_fn=collate_single)` usable
    directly with every training script in `pharmaforge_core.training`.
    """
    assert len(batch) == 1, "PharmaForge training scripts use batch_size=1 (see collate_single docstring)"
    return batch[0]


class SyntheticPocketLigandDataset(Dataset):
    """Randomly generated pocket/ligand tensor pairs with the exact same
    shapes/dtypes real data would produce. Used by unit tests and CI so the
    full training loop, loss computation and checkpointing logic can be
    exercised without a multi-GB dataset download.
    """

    def __init__(self, n_samples: int = 16, n_pocket_atoms: int = 30, n_ligand_atoms: int = 12, seed: int = 0):
        self.n_samples = n_samples
        self.n_pocket_atoms = n_pocket_atoms
        self.n_ligand_atoms = n_ligand_atoms
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        from pharmaforge_core.config import ATOM_VOCAB, RESIDUE_VOCAB
        from pharmaforge_core.data.preprocessing import PHYSCHEM_DIM

        g = self.generator
        n_p, n_l = self.n_pocket_atoms, self.n_ligand_atoms
        pocket = {
            "atom_type_idx": torch.randint(1, len(ATOM_VOCAB) - 1, (n_p,), generator=g),
            "residue_type_idx": torch.randint(1, len(RESIDUE_VOCAB) - 1, (n_p,), generator=g),
            "physchem": torch.randn(n_p, PHYSCHEM_DIM, generator=g),
            "coords": torch.randn(n_p, 3, generator=g) * 8.0,
        }
        ligand = {
            "atom_type_idx": torch.randint(1, len(ATOM_VOCAB) - 1, (n_l,), generator=g),
            "physchem": torch.randn(n_l, PHYSCHEM_DIM, generator=g),
            "coords": torch.randn(n_l, 3, generator=g) * 3.0,
        }
        return {"complex_id": f"synthetic_{idx}", "pocket": pocket, "ligand": ligand}
