"""3D conditional diffusion model for pocket-conditioned de novo generation.

Replaces AlphaDrug's SMILES-token autoregressive decoder with a denoising
diffusion probabilistic model (DDPM, Ho et al. 2020) operating jointly over:

  * ligand atom coordinates (continuous, R^3)      -> Gaussian diffusion
  * ligand atom types       (categorical, K types) -> logits regressed by
    the same network and decoded greedily/temperature-sampled at each step

The pocket graph (fixed coordinates + residue/physchem features) is passed
as conditioning context through `PocketConditionedEGNN`, exactly like the
scoring stack and the actor-critic, so all three modules share one notion
of "what does this binding site look like".

This is the generative half of the pipeline described in the project
whitepaper (docs/whitepaper.md); `pharmaforge_core.generation.sampler` wraps it
with classifier-guidance from the scoring stack and with the
selectivity-constrained sampling loop.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pharmaforge_core.config import ATOM_VOCAB, DiffusionConfig
from pharmaforge_core.models.egnn import PocketConditionedEGNN
from pharmaforge_core.utils.geometry import radius_graph


def make_beta_schedule(cfg: DiffusionConfig) -> Tensor:
    t = torch.arange(cfg.n_timesteps + 1, dtype=torch.float64) / cfg.n_timesteps
    if cfg.schedule == "cosine":
        s = 0.008
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f / f[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clamp(betas, 1e-8, 0.999).float()
    return torch.linspace(cfg.beta_start, cfg.beta_end, cfg.n_timesteps).float()


class GaussianDiffusionSchedule:
    """Precomputed noise-schedule constants, shared by q(x_t | x_0) and the
    reverse denoising step p(x_{t-1} | x_t)."""

    def __init__(self, cfg: DiffusionConfig):
        self.cfg = cfg
        self.betas = make_beta_schedule(cfg)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device):
        for name in (
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod", "posterior_variance",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Forward noising: x_t = sqrt(a_bar_t) x0 + sqrt(1 - a_bar_t) eps."""
        a = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (x0.dim() - 1)))
        b = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x0.dim() - 1)))
        return a * x0 + b * noise


class LigandDenoiser(nn.Module):
    """epsilon-prediction network: given noisy ligand coords x_t and the
    fixed pocket context, predict the noise added to the coordinates and the
    clean-data logits over atom types.
    """

    def __init__(self, cfg: DiffusionConfig, physchem_dim: int = 6):
        super().__init__()
        self.cfg = cfg
        self.backbone = PocketConditionedEGNN(cfg.egnn, physchem_dim=physchem_dim, time_embed=True)
        self.atom_type_head = nn.Sequential(
            nn.Linear(cfg.egnn.hidden_dim, cfg.egnn.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.egnn.hidden_dim, len(ATOM_VOCAB)),
        )

    def forward(
        self,
        pocket_atom_type_idx: Tensor,
        pocket_residue_type_idx: Tensor,
        pocket_physchem: Tensor,
        pocket_coords: Tensor,
        ligand_atom_type_idx: Tensor,
        ligand_physchem: Tensor,
        ligand_coords_t: Tensor,
        timestep: Tensor,
        cutoff: float = 10.0,
        max_neighbors: int = 32,
    ) -> tuple[Tensor, Tensor]:
        n_pocket = pocket_coords.shape[0]
        n_ligand = ligand_coords_t.shape[0]
        coords = torch.cat([pocket_coords, ligand_coords_t], dim=0)
        atom_type_idx = torch.cat([pocket_atom_type_idx, ligand_atom_type_idx], dim=0)
        residue_type_idx = torch.cat(
            [pocket_residue_type_idx, torch.zeros(n_ligand, dtype=torch.long, device=coords.device)], dim=0
        )
        physchem = torch.cat([pocket_physchem, ligand_physchem], dim=0)
        is_ligand = torch.cat(
            [torch.zeros(n_pocket, dtype=torch.long), torch.ones(n_ligand, dtype=torch.long)]
        ).to(coords.device)

        edge_index = radius_graph(coords, cutoff=cutoff, max_neighbors=max_neighbors)
        h, x_out = self.backbone(
            atom_type_idx, residue_type_idx, is_ligand, physchem, coords, edge_index, timestep=timestep
        )
        ligand_h = h[n_pocket:]
        ligand_x_out = x_out[n_pocket:]
        predicted_noise = ligand_x_out - ligand_coords_t  # backbone predicts the denoised displacement
        atom_type_logits = self.atom_type_head(ligand_h)
        return predicted_noise, atom_type_logits


class LigandDiffusionModel(nn.Module):
    """Ties `LigandDenoiser` + `GaussianDiffusionSchedule` together and
    exposes `training_loss` and `p_sample_loop` (ancestral sampling with
    optional classifier-style guidance from an external scoring function).
    """

    def __init__(self, cfg: DiffusionConfig, physchem_dim: int = 6):
        super().__init__()
        self.cfg = cfg
        self.denoiser = LigandDenoiser(cfg, physchem_dim=physchem_dim)
        self.schedule = GaussianDiffusionSchedule(cfg)

    def to(self, device):
        super().to(device)
        self.schedule.to(device)
        return self

    def training_loss(
        self,
        pocket: dict,
        ligand_coords_0: Tensor,
        ligand_atom_type_idx: Tensor,
        ligand_physchem: Tensor,
    ) -> Tensor:
        device = ligand_coords_0.device
        t = torch.randint(0, self.cfg.n_timesteps, (1,), device=device)
        noise = torch.randn_like(ligand_coords_0)
        # Center noise to remove the translational DOF, standard practice
        # for equivariant molecule diffusion (Hoogeboom et al., E(3) diffusion).
        noise = noise - noise.mean(dim=0, keepdim=True)
        x_t = self.schedule.q_sample(ligand_coords_0, t, noise)

        pred_noise, atom_type_logits = self.denoiser(
            pocket["atom_type_idx"], pocket["residue_type_idx"], pocket["physchem"], pocket["coords"],
            ligand_atom_type_idx, ligand_physchem, x_t, t.float(),
        )
        coord_loss = F.mse_loss(pred_noise, noise)
        type_loss = F.cross_entropy(atom_type_logits, ligand_atom_type_idx)
        return coord_loss + type_loss

    def p_sample_loop(
        self,
        pocket: dict,
        n_ligand_atoms: int,
        guidance_fn=None,
        guidance_scale: float | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        """Ancestral sampling from pure noise down to x_0.

        `guidance_fn(coords, atom_type_idx) -> scalar score` lets the caller
        (see generation/sampler.py) plug in the differentiable scoring stack
        as a classifier-style gradient guidance term, and/or the
        selectivity-profiler's anti-target penalty, without changing this loop.
        """
        cfg = self.cfg
        guidance_scale = guidance_scale if guidance_scale is not None else cfg.guidance_scale
        x_t = torch.randn(n_ligand_atoms, 3, device=device)
        x_t = x_t - x_t.mean(dim=0, keepdim=True)
        atom_type_idx = torch.randint(1, len(ATOM_VOCAB) - 1, (n_ligand_atoms,), device=device)
        ligand_physchem = torch.zeros(n_ligand_atoms, pocket["physchem"].shape[-1], device=device)

        for step in reversed(range(cfg.n_timesteps)):
            t = torch.tensor([step], device=device, dtype=torch.float32)
            beta_t = self.schedule.betas[step]
            alpha_t = self.schedule.alphas[step]
            alpha_bar_t = self.schedule.alphas_cumprod[step]

            if guidance_fn is not None:
                with torch.enable_grad():
                    x_t = x_t.detach().requires_grad_(True)
                    pred_noise, atom_type_logits = self.denoiser(
                        pocket["atom_type_idx"], pocket["residue_type_idx"], pocket["physchem"], pocket["coords"],
                        atom_type_idx, ligand_physchem, x_t, t,
                    )
                    score = guidance_fn(x_t, atom_type_idx)
                    grad = torch.autograd.grad(score.sum(), x_t, retain_graph=False)[0]
                pred_noise = (pred_noise - guidance_scale * torch.sqrt(1 - alpha_bar_t) * grad).detach()
                x_t = x_t.detach()
                atom_type_logits = atom_type_logits.detach()
            else:
                with torch.no_grad():
                    pred_noise, atom_type_logits = self.denoiser(
                        pocket["atom_type_idx"], pocket["residue_type_idx"], pocket["physchem"], pocket["coords"],
                        atom_type_idx, ligand_physchem, x_t, t,
                    )

            with torch.no_grad():
                mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise)
                if step > 0:
                    noise = torch.randn_like(x_t)
                    noise = noise - noise.mean(dim=0, keepdim=True)
                    x_t = mean + torch.sqrt(self.schedule.posterior_variance[step]) * noise
                else:
                    x_t = mean
                atom_type_idx = atom_type_logits.argmax(dim=-1)

        return x_t.detach(), atom_type_idx.detach()
