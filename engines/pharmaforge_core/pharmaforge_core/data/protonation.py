"""Protonation state assignment via PDB2PQR.

Both training-set preprocessing (PDBbind/CrossDocked2020 raw structures) and
the interactive dashboard need consistent protonation before pocket atoms
are fed to the EGNN backbone, since H-bond donor/acceptor features depend on
which polar hydrogens are actually present at physiological pH.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def protonate_structure(pdb_path: Path, output_path: Path | None = None, ph: float = 7.4) -> Path:
    """Run `pdb2pqr30` to add hydrogens and assign partial charges.

    Falls back to returning the input path unmodified (with a warning) if
    pdb2pqr isn't importable/executable in this environment, so the rest of
    the pipeline degrades gracefully rather than hard-failing - heavy-atom
    features (element, residue type, charge) are still usable without
    explicit hydrogens, at the cost of less precise H-bond geometry.
    """
    output_path = output_path or pdb_path.with_suffix(".pqr.pdb")
    pdb2pqr_bin = shutil.which("pdb2pqr30") or shutil.which("pdb2pqr")
    if pdb2pqr_bin is None:
        try:
            from pdb2pqr.main import main as pdb2pqr_main  # type: ignore

            pdb2pqr_main(
                [
                    "--ff=PARSE",
                    f"--with-ph={ph}",
                    "--titration-state-method=propka",
                    str(pdb_path),
                    str(output_path),
                ]
            )
            return output_path
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("pdb2pqr unavailable (%s); skipping explicit protonation", exc)
            return pdb_path

    subprocess.run(
        [pdb2pqr_bin, "--ff=PARSE", f"--with-ph={ph}", str(pdb_path), str(output_path)],
        check=True,
        capture_output=True,
    )
    return output_path
