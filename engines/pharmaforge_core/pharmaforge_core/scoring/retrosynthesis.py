"""Retrosynthesis-based synthetic accessibility via AiZynthFinder.

AiZynthFinder (Genheden et al., 2020) needs a trained one-step retrosynthesis
policy + a stock/building-block file, both multi-GB downloads not vendored
in this repository (see `scripts/download_pretrained.py` for the fetch
command). This wrapper degrades gracefully to the RDKit `SAscore` heuristic
(`utils.chem.synthetic_accessibility_score`) when AiZynthFinder or its data
files aren't present, so `analysis/physchem.py` always returns a synthetic
accessibility number - just a coarser one - without a hard dependency.
"""
from __future__ import annotations

import logging
from pathlib import Path

from rdkit import Chem

from pharmaforge_core.utils.chem import synthetic_accessibility_score

logger = logging.getLogger(__name__)


class RetrosynthesisScorer:
    def __init__(self, config_path: str | Path | None = None):
        self.config_path = Path(config_path) if config_path else None
        self._finder = None
        if self.config_path and self.config_path.exists():
            try:
                from aizynthfinder.aizynthfinder import AiZynthFinder  # type: ignore

                self._finder = AiZynthFinder(configfile=str(self.config_path))
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.warning("AiZynthFinder unavailable (%s); falling back to SAscore heuristic", exc)

    def score(self, mol: Chem.Mol) -> dict:
        smiles = Chem.MolToSmiles(mol)
        if self._finder is None:
            return {
                "method": "sa_score_fallback",
                "sa_score": synthetic_accessibility_score(mol),
                "route_found": None,
                "n_steps": None,
            }

        self._finder.target_smiles = smiles
        self._finder.tree_search()
        self._finder.build_routes()
        stats = self._finder.extract_statistics()
        solved = bool(stats.get("is_solved", False))
        return {
            "method": "aizynthfinder",
            "sa_score": synthetic_accessibility_score(mol),
            "route_found": solved,
            "n_steps": stats.get("number_of_steps") if solved else None,
        }
