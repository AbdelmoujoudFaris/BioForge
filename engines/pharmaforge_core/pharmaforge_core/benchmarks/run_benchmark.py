"""Runs PharmaForge's own generation + analysis pipeline over a set of test
pockets and reports the same metric set (Vina-like score, QED, SA,
diversity, novelty) used by `baselines.py`'s literature reference table, so
the two can be placed side by side.

    python -m pharmaforge_core.benchmarks.run_benchmark --pdb-ids 6LU7 3PBL 4YHJ \\
        --checkpoint-dir checkpoints --n-candidates-per-pocket 20

Without `--checkpoint-dir` pointing at real trained weights, this reports
PharmaForge's numbers for an **untrained** model - a useful sanity check
that the harness runs end-to-end, but not a meaningful comparison against
the trained baselines in `baselines.py`. The script prints a loud warning in
that case and labels the output row accordingly; it does not fabricate a
"PharmaForge outperforms X" conclusion for you.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from pharmaforge_core.analysis.diversity import internal_diversity
from pharmaforge_core.analysis.physchem import profile_candidates, summarize
from pharmaforge_core.benchmarks.baselines import reference_table_markdown
from pharmaforge_core.config import DiffusionConfig, EGNNConfig, GenerationConfig, ScoringConfig
from pharmaforge_core.data.pdb_download import fetch_structure
from pharmaforge_core.data.pocket_extraction import (
    extract_pocket_around_point,
    find_fpocket_cavities,
    geometric_center_fallback,
)
from pharmaforge_core.data.preprocessing import pocket_to_tensors
from pharmaforge_core.generation.sampler import GuidedDiffusionSampler
from pharmaforge_core.models.diffusion import LigandDiffusionModel
from pharmaforge_core.models.scoring import ScoringStack

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_or_init_models(checkpoint_dir: str | None, device: str, hidden_dim: int = 128, n_layers: int = 6):
    egnn_cfg = EGNNConfig(node_feature_dim=hidden_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    diffusion_model = LigandDiffusionModel(DiffusionConfig(egnn=egnn_cfg)).to(device)
    scoring_stack = ScoringStack(ScoringConfig(egnn=egnn_cfg)).to(device)

    loaded = False
    if checkpoint_dir:
        diff_ckpt = Path(checkpoint_dir) / "diffusion.pt"
        score_ckpt = Path(checkpoint_dir) / "scoring.pt"
        if diff_ckpt.exists():
            diffusion_model.load_state_dict(torch.load(diff_ckpt, map_location=device))
            loaded = True
        if score_ckpt.exists():
            state = torch.load(score_ckpt, map_location=device)
            scoring_stack.load_state_dict(state.get("model_state_dict", state))
            loaded = True
    diffusion_model.eval()
    scoring_stack.eval()
    return diffusion_model, scoring_stack, loaded


def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    diffusion_model, scoring_stack, loaded_checkpoint = load_or_init_models(
        args.checkpoint_dir, device, hidden_dim=args.hidden_dim, n_layers=args.n_layers
    )
    if not loaded_checkpoint:
        logger.warning(
            "*** No trained checkpoint found - running the benchmark against RANDOM WEIGHTS. ***\n"
            "This validates the harness end-to-end but the resulting metrics are NOT a real\n"
            "comparison against Pocket2Mol/DiffSBDD/TargetDiff/AlphaDrug. Train first (see\n"
            "pharmaforge_core/training/) and re-run with --checkpoint-dir pointing at real weights."
        )

    sampler = GuidedDiffusionSampler(diffusion_model, scoring_stack, device=device)
    gen_cfg = GenerationConfig(
        n_candidates=args.n_candidates_per_pocket,
        max_atoms=args.max_atoms,
        diffusion=DiffusionConfig(n_timesteps=args.n_timesteps),
    )

    all_mols, all_scores = [], []
    for pdb_id in args.pdb_ids:
        logger.info("generating for pocket %s", pdb_id)
        pdb_path = fetch_structure(pdb_id)
        centers = find_fpocket_cavities(pdb_path)
        center = centers[0] if centers else geometric_center_fallback(pdb_path)
        pocket = extract_pocket_around_point(pdb_path, center)
        pocket_tensors = pocket_to_tensors(pocket, device=device)

        candidates = sampler.generate(pocket_tensors, gen_cfg, target_name=pdb_id)
        all_mols.extend(c.mol_or_none for c in candidates)
        all_scores.extend(c.score for c in candidates)

    physchem_df = profile_candidates(all_mols)
    summary = summarize(physchem_df)
    valid_mols = [m for m in all_mols if m is not None]

    pharmaforge_row = {
        "vina_score": None,  # differentiable_score is a learned surrogate, not SMINA/Vina - report separately, don't conflate units
        "learned_score_proxy": float(np.mean(all_scores)) if all_scores else None,
        "qed": summary.get("mean_qed"),
        "sa": summary.get("mean_sa_score"),
        "diversity": internal_diversity(valid_mols) if valid_mols else None,
        "validity_rate": summary.get("validity_rate"),
        "source": "this run" + ("" if loaded_checkpoint else " (UNTRAINED WEIGHTS - not comparable)"),
    }

    print("\n=== PharmaForge (this run) ===")
    for k, v in pharmaforge_row.items():
        print(f"  {k}: {v}")

    print("\n=== Literature reference table (see baselines.py caveats) ===")
    print(reference_table_markdown())

    return pharmaforge_row


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdb-ids", nargs="+", default=["6LU7"])
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--n-candidates-per-pocket", type=int, default=20)
    p.add_argument("--max-atoms", type=int, default=25)
    p.add_argument("--n-timesteps", type=int, default=200)
    p.add_argument("--hidden-dim", type=int, default=128, help="EGNN hidden dim; shrink for a fast CPU demo run")
    p.add_argument("--n-layers", type=int, default=6, help="EGNN layer count; shrink for a fast CPU demo run")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
