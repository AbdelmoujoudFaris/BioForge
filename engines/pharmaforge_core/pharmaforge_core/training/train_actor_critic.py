"""REINFORCE-style training of the `GraphTransformerActor` against a frozen,
pretrained `ScoringStack` critic (see `train_scoring.py` for how that
checkpoint is produced).

Each episode: start from a random single-atom seed in a sampled pocket, let
the actor grow the molecule atom-by-atom up to `--max-atoms`, score the
final molecule with the critic, and update the actor with the policy
gradient `-log_prob * (final_score - baseline)`, where `baseline` is a
running mean of final scores (a simple variance-reduction trick, standard in
REINFORCE - avoids needing a separate learned value head since the critic
already gives an exact terminal reward).

    python -m pharmaforge_core.training.train_actor_critic \\
        --scoring-checkpoint checkpoints/scoring/scoring.pt --episodes 5000
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from tqdm import trange

from pharmaforge_core.config import ActorCriticConfig, ScoringConfig
from pharmaforge_core.data.datasets import SyntheticPocketLigandDataset
from pharmaforge_core.models.actor_critic import ActorCritic
from pharmaforge_core.models.scoring import ScoringStack
from pharmaforge_core.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_episode(actor_critic: ActorCritic, pocket: dict, max_atoms: int, temperature: float):
    device = pocket["coords"].device
    atom_type_idx = torch.randint(1, 11, (1,), device=device)
    physchem = torch.zeros(1, pocket["physchem"].shape[-1], device=device)
    coords = torch.zeros(1, 3, device=device)

    log_probs = []
    for _ in range(max_atoms - 1):
        action = actor_critic.actor.sample_action(pocket, atom_type_idx, physchem, coords, temperature)
        if action["new_atom_type"].item() == 0:  # STOP
            break
        log_probs.append(action["log_prob"])

        attach_point = coords[action["attach_idx"].item()]
        new_coord = attach_point + torch.randn(3, device=device) * 1.5
        atom_type_idx = torch.cat([atom_type_idx, action["new_atom_type"].unsqueeze(0)])
        physchem = torch.cat([physchem, physchem.mean(dim=0, keepdim=True)])
        coords = torch.cat([coords, new_coord.unsqueeze(0)])

    with torch.no_grad():
        final_score = actor_critic.critic.differentiable_score(pocket, atom_type_idx, physchem, coords)
    return log_probs, final_score, coords.shape[0]


def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    scoring_cfg = ScoringConfig()
    critic = ScoringStack(scoring_cfg).to(device)
    if args.scoring_checkpoint and Path(args.scoring_checkpoint).exists():
        state = torch.load(args.scoring_checkpoint, map_location=device)
        critic.load_state_dict(state.get("model_state_dict", state))
        logger.info("loaded critic checkpoint from %s", args.scoring_checkpoint)
    else:
        logger.warning("no scoring checkpoint found; training actor against an UNTRAINED critic (demo only)")
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)

    actor_critic = ActorCritic(ActorCriticConfig(egnn=scoring_cfg.egnn), critic).to(device)
    optimizer = torch.optim.AdamW(actor_critic.actor.parameters(), lr=args.lr)

    pocket_source = SyntheticPocketLigandDataset(n_samples=args.n_pockets)
    baseline = 0.0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for episode in trange(args.episodes, desc="episodes"):
        sample = pocket_source[episode % len(pocket_source)]
        pocket = {k: v.to(device) for k, v in sample["pocket"].items()}

        log_probs, final_score, n_atoms = run_episode(actor_critic, pocket, args.max_atoms, args.temperature)
        if not log_probs:
            continue

        advantage = final_score.item() - baseline
        baseline = 0.95 * baseline + 0.05 * final_score.item()

        policy_loss = -torch.stack(log_probs).sum() * advantage
        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_critic.actor.parameters(), max_norm=1.0)
        optimizer.step()

        if (episode + 1) % args.log_every == 0:
            logger.info(
                "episode %d score=%.3f baseline=%.3f n_atoms=%d", episode + 1, final_score.item(), baseline, n_atoms
            )
        if (episode + 1) % args.save_every == 0 or episode == args.episodes - 1:
            torch.save(actor_critic.actor.state_dict(), checkpoint_dir / "actor.pt")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scoring-checkpoint", type=str, default="checkpoints/scoring/scoring.pt")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--n-pockets", type=int, default=32, help="synthetic pocket pool size (replace with a real dataset for production training)")
    p.add_argument("--max-atoms", type=int, default=25)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/actor_critic")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
