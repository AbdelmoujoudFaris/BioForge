"""Binding-pocket extraction from a receptor structure.

Two modes:

  * `extract_pocket_around_ligand` - carve out every receptor atom within
    `radius_around_ligand` angstrom of a reference co-crystallized ligand.
    This is the mode used for training (PDBbind/CrossDocked2020 both ship a
    reference ligand per complex) and for "grow from this seed" generation.

  * `extract_pocket_fpocket` - shell out to the `fpocket` cavity-detection
    binary for blind pocket discovery when no reference ligand is given
    (e.g. a fresh PDB ID typed into the dashboard with an apo structure).
    Falls back to a geometric-center heuristic if the `fpocket` executable
    isn't on PATH, so the rest of the pipeline still runs without requiring
    every contributor to compile fpocket locally.

Both paths return a plain `Pocket` dataclass consumed by
`preprocessing.py` to build the tensors the EGNN backbone expects.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB import NeighborSearch, PDBParser
from Bio.PDB.Structure import Structure

from pharmaforge_core.config import PocketConfig

logger = logging.getLogger(__name__)

_HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "PHE", "MET", "TRP", "PRO"}
_HBOND_DONOR_RESIDUES = {"SER", "THR", "TYR", "ASN", "GLN", "LYS", "ARG", "HIS", "TRP", "CYS"}
_HBOND_ACCEPTOR_RESIDUES = {"SER", "THR", "TYR", "ASN", "GLN", "ASP", "GLU", "HIS"}
_CHARGE = {"ASP": -1.0, "GLU": -1.0, "LYS": 1.0, "ARG": 1.0, "HIS": 0.1}


@dataclass
class Pocket:
    elements: list[str]
    residue_names: list[str]
    coords: np.ndarray  # (N, 3)
    is_hydrophobic: np.ndarray
    is_hbond_donor: np.ndarray
    is_hbond_acceptor: np.ndarray
    charge: np.ndarray
    source_pdb_id: str
    center: np.ndarray


def _residue_physchem_row(resname: str) -> tuple[bool, bool, bool, float]:
    return (
        resname in _HYDROPHOBIC_RESIDUES,
        resname in _HBOND_DONOR_RESIDUES,
        resname in _HBOND_ACCEPTOR_RESIDUES,
        _CHARGE.get(resname, 0.0),
    )


def _load_structure(pdb_path: Path) -> Structure:
    parser = PDBParser(QUIET=True)
    return parser.get_structure(pdb_path.stem, str(pdb_path))


def extract_pocket_around_ligand(
    pdb_path: Path,
    ligand_resname: str,
    cfg: PocketConfig | None = None,
) -> Pocket:
    cfg = cfg or PocketConfig()
    structure = _load_structure(pdb_path)
    ligand_atoms = [
        atom
        for residue in structure.get_residues()
        if residue.get_resname() == ligand_resname
        for atom in residue
    ]
    if not ligand_atoms:
        raise ValueError(f"Ligand residue '{ligand_resname}' not found in {pdb_path.name}")

    protein_atoms = [
        atom
        for residue in structure.get_residues()
        if residue.get_resname() != ligand_resname and residue.id[0] == " "
        for atom in residue
        if cfg.include_hydrogens or atom.element != "H"
    ]
    ns = NeighborSearch(protein_atoms)

    seen = {}
    for lig_atom in ligand_atoms:
        for atom in ns.search(lig_atom.coord, cfg.radius_around_ligand):
            seen[atom.get_serial_number()] = atom
    pocket_atoms = list(seen.values())[: cfg.max_pocket_atoms]
    ligand_center = np.mean([a.coord for a in ligand_atoms], axis=0)

    return _atoms_to_pocket(pocket_atoms, pdb_path.stem, ligand_center)


def extract_pocket_around_point(
    pdb_path: Path,
    center: np.ndarray,
    cfg: PocketConfig | None = None,
) -> Pocket:
    """Blind-pocket fallback: carve every protein atom within
    `radius_around_centroid` of a given 3D point (typically an fpocket
    cavity center, see `extract_pocket_fpocket`).
    """
    cfg = cfg or PocketConfig()
    structure = _load_structure(pdb_path)
    protein_atoms = [
        atom
        for residue in structure.get_residues()
        if residue.id[0] == " "
        for atom in residue
        if cfg.include_hydrogens or atom.element != "H"
    ]
    ns = NeighborSearch(protein_atoms)
    pocket_atoms = ns.search(center, cfg.radius_around_centroid)[: cfg.max_pocket_atoms]
    return _atoms_to_pocket(pocket_atoms, pdb_path.stem, center)


def _atoms_to_pocket(pocket_atoms: list, source_id: str, center: np.ndarray) -> Pocket:
    elements, residue_names, coords = [], [], []
    hydrophobic, donor, acceptor, charge = [], [], [], []
    for atom in pocket_atoms:
        residue = atom.get_parent()
        resname = residue.get_resname()
        elements.append(atom.element.capitalize() if atom.element else "C")
        residue_names.append(resname)
        coords.append(atom.coord)
        hp, hd, ha, ch = _residue_physchem_row(resname)
        hydrophobic.append(hp)
        donor.append(hd)
        acceptor.append(ha)
        charge.append(ch)

    return Pocket(
        elements=elements,
        residue_names=residue_names,
        coords=np.array(coords, dtype=np.float32),
        is_hydrophobic=np.array(hydrophobic, dtype=bool),
        is_hbond_donor=np.array(donor, dtype=bool),
        is_hbond_acceptor=np.array(acceptor, dtype=bool),
        charge=np.array(charge, dtype=np.float32),
        source_pdb_id=source_id,
        center=np.asarray(center, dtype=np.float32),
    )


def find_fpocket_cavities(pdb_path: Path) -> list[np.ndarray]:
    """Run the `fpocket` binary (must be installed separately; see
    docs/architecture.md) and parse cavity centers from its output pockets.
    Returns an empty list (caller should fall back to the structure's
    geometric center) if fpocket isn't available - this keeps the pipeline
    runnable in CI/dev environments that skip the external binary.
    """
    fpocket_bin = shutil.which("fpocket")
    if fpocket_bin is None:
        logger.warning("fpocket not found on PATH; falling back to geometric-center pocket detection")
        return []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / pdb_path.name
        tmp_path.write_bytes(pdb_path.read_bytes())
        subprocess.run([fpocket_bin, "-f", str(tmp_path)], check=True, capture_output=True)
        out_dir = tmp_path.parent / f"{tmp_path.stem}_out" / "pockets"
        centers = []
        if out_dir.exists():
            for pqr_file in sorted(out_dir.glob("pocket*_atm.pdb")):
                structure = _load_structure(pqr_file)
                coords = np.array([atom.coord for atom in structure.get_atoms()])
                centers.append(coords.mean(axis=0))
        return centers


def geometric_center_fallback(pdb_path: Path) -> np.ndarray:
    structure = _load_structure(pdb_path)
    coords = np.array([atom.coord for atom in structure.get_atoms() if atom.element != "H"])
    return coords.mean(axis=0)
