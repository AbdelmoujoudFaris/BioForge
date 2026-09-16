"""Trains `LigandDiffusionModel` on a PDBbind- or CrossDocked2020-style
complex directory (see `data.datasets`).

    python -m pharmaforge_core.training.train_diffusion \\
        --data-root /path/to/CrossDocked2020 --epochs 100 \\
        --checkpoint-dir checkpoints/diffusion

Requires a real, downloaded copy of PDBbind or CrossDocked2020 and a GPU for
a full-scale run; `--synthetic` runs the identical loop against
`SyntheticPocketLigandDataset` so the training mechanics (loss, optimizer,
checkpointing, logging) can be smoke-tested in CI without either.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pharmaforge_core.config import DiffusionConfig
from pharmaforge_core.data.datasets import (
    CrossDocked2020Dataset,
    PDBBindDataset,
    SyntheticPocketLigandDataset,
    collate_single,
)
from pharmaforge_core.models.diffusion import LigandDiffusionModel
from pharmaforge_core.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_dataset(args):
    if args.synthetic:
        return SyntheticPocketLigandDataset(n_samples=args.synthetic_samples)
    if args.dataset == "pdbbind":
        return PDBBindDataset(args.data_root)
    return CrossDocked2020Dataset(args.data_root)


def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    dataset = build_dataset(args)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_single)

    model = LigandDiffusionModel(DiffusionConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch}"):
            pocket = {k: v.to(device) for k, v in batch["pocket"].items()}
            ligand = {k: v.to(device) for k, v in batch["ligand"].items()}

            optimizer.zero_grad()
            loss = model.training_loss(pocket, ligand["coords"], ligand["atom_type_idx"], ligand["physchem"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / max(len(loader), 1)
        logger.info("epoch %d avg_loss=%.4f", epoch, avg_loss)

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = checkpoint_dir / f"diffusion_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), checkpoint_dir / "diffusion.pt")
            logger.info("saved checkpoint to %s", ckpt_path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--dataset", choices=["pdbbind", "crossdocked2020"], default="crossdocked2020")
    p.add_argument("--synthetic", action="store_true", help="train against synthetic random data (smoke test / CI)")
    p.add_argument("--synthetic-samples", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/diffusion")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
