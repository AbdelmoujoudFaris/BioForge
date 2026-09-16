"""Geometric primitives shared by the EGNN backbone and the analysis suite.

Every function here is written to be batch- and autograd-friendly since the
same code path is used inside the diffusion denoiser (needs gradients) and
inside the offline analysis scripts (does not).
"""
from __future__ import annotations

import torch
from torch import Tensor


def pairwise_distances(coords: Tensor) -> Tensor:
    """Euclidean distance matrix. coords: (N, 3) -> (N, N)."""
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    return torch.linalg.norm(diff, dim=-1)


def radius_graph(
    coords: Tensor,
    cutoff: float,
    max_neighbors: int = 32,
    loop: bool = False,
) -> Tensor:
    """Build a k-nn-capped radius graph.

    Returns edge_index of shape (2, E) in COO format (source, target).
    Capping neighbors keeps the graph sparse for large pockets (hundreds of
    atoms) without needing a spatial index library.

    Fully vectorized (a single `torch.cdist` + masked `topk`, no Python loop
    over atoms) rather than PyTorch Geometric's own `radius_graph`, which
    needs the optional `torch-cluster`/`pyg-lib` compiled extensions -
    prebuilt wheels for those lag behind new PyTorch releases and often
    require building from source, which is unnecessary friction for a graph
    this small (at most a few hundred pocket+ligand atoms).
    """
    n = coords.shape[0]
    dist = pairwise_distances(coords)
    if not loop:
        dist.fill_diagonal_(float("inf"))

    k = min(max_neighbors, n - (0 if loop else 1))
    if k <= 0:
        return torch.zeros((2, 0), dtype=torch.long, device=coords.device)

    neg_dist, neighbor_idx = torch.topk(-dist, k=k, dim=1)
    within_cutoff = (-neg_dist) <= cutoff

    source = torch.arange(n, device=coords.device).unsqueeze(1).expand(-1, k)[within_cutoff]
    target = neighbor_idx[within_cutoff]
    return torch.stack([source, target], dim=0).long()


def kabsch_rmsd(p: Tensor, q: Tensor) -> float:
    """Optimal-superposition RMSD between two point sets of equal size.

    Used to score generated poses against a known co-crystallized ligand
    (structural validation metric), not as a differentiable training loss.
    """
    p = p - p.mean(dim=0, keepdim=True)
    q = q - q.mean(dim=0, keepdim=True)
    cov = p.t() @ q
    u, _, vt = torch.linalg.svd(cov)
    d = torch.sign(torch.linalg.det(vt.t() @ u.t()))
    correction = torch.eye(3, device=p.device)
    correction[-1, -1] = d
    rot = vt.t() @ correction @ u.t()
    p_aligned = p @ rot.t()
    return torch.sqrt(torch.mean(torch.sum((p_aligned - q) ** 2, dim=-1))).item()


def center_of_mass(coords: Tensor, masses: Tensor | None = None) -> Tensor:
    if masses is None:
        return coords.mean(dim=0)
    masses = masses.unsqueeze(-1)
    return (coords * masses).sum(dim=0) / masses.sum()


def random_rotation_matrix(device=None) -> Tensor:
    """Uniformly sampled 3D rotation, used for E(3)-equivariance unit tests."""
    q = torch.randn(4, device=device)
    q = q / torch.linalg.norm(q)
    w, x, y, z = q
    return torch.tensor(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ],
        device=device,
    )


def clash_score(coords: Tensor, vdw_radii: Tensor, tolerance: float = 0.4) -> Tensor:
    """Soft steric clash penalty: sum of overlap beyond `tolerance` * (r_i + r_j).

    Differentiable, used both as an auxiliary loss during diffusion training
    and as one head's ground-truth target in the scoring stack.
    """
    dist = pairwise_distances(coords)
    radii_sum = vdw_radii.unsqueeze(0) + vdw_radii.unsqueeze(1)
    threshold = radii_sum * (1.0 - tolerance)
    overlap = torch.clamp(threshold - dist, min=0.0)
    n = coords.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=coords.device)
    return (overlap * mask).sum() / 2.0
