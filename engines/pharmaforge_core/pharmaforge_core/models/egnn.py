"""E(3)-equivariant graph neural network backbone.

Implements the EGNN message-passing rule of Satorras et al. (2021),
"E(n) Equivariant Graph Neural Networks", specialized for joint
pocket + ligand graphs:

  * Pocket atoms carry fixed 3D coordinates (the rigid receptor context)
    and are never displaced.
  * Ligand atoms carry coordinates that are updated at every layer -
    this is what lets the same backbone serve as (a) the denoiser in the
    3D diffusion generator, (b) the actor's per-atom action head, and
    (c) the scoring stack's pose encoder, all conditioned on identical
    pocket geometry.

Rotation/translation equivariance of the coordinate channel plus
permutation invariance of the node/edge features is what replaces
AlphaDrug's 1D SMILES-token Lmser Transformer, which has no notion of
3D structure at all.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

from pharmaforge_core.config import EGNNConfig


def sinusoidal_time_embedding(t: Tensor, dim: int) -> Tensor:
    """Standard transformer-style sinusoidal embedding of diffusion timestep t."""
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, device=device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / max(half - 1, 1))
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class EGNNLayer(MessagePassing):
    """One equivariant message-passing block.

    Node feature update:  h_i' = h_i + phi_h(h_i, agg_j m_ij)
    Coordinate update:    x_i' = x_i + agg_j (x_i - x_j) * phi_x(m_ij)   [only for updatable nodes]
    Edge message:         m_ij = phi_e(h_i, h_j, ||x_i - x_j||^2, e_ij)
    """

    def __init__(self, hidden_dim: int, edge_feature_dim: int = 0):
        super().__init__(aggr="add", node_dim=0)
        self.edge_feature_dim = edge_feature_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        # Small init on the last coordinate layer stabilizes early training,
        # a trick carried over from the original EGNN reference implementation.
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.01)

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        update_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """update_mask: (N,) bool, True for nodes whose coordinates may move
        (ligand atoms). Pocket atoms are excluded so the receptor stays rigid.
        """
        if edge_attr is None and self.edge_feature_dim > 0:
            edge_attr = h.new_zeros(edge_index.shape[1], self.edge_feature_dim)
        h_out, coord_delta = self.propagate(edge_index, h=h, x=x, edge_attr=edge_attr)
        h_new = h + self.node_mlp(torch.cat([h, h_out], dim=-1))
        if update_mask is None:
            x_new = x + coord_delta
        else:
            x_new = x + coord_delta * update_mask.unsqueeze(-1).float()
        return h_new, x_new

    def message(self, h_i: Tensor, h_j: Tensor, x_i: Tensor, x_j: Tensor, edge_attr: Tensor | None):
        rel = x_i - x_j
        dist2 = (rel**2).sum(dim=-1, keepdim=True)
        feats = [h_i, h_j, dist2]
        if edge_attr is not None:
            feats.append(edge_attr)
        m_ij = self.edge_mlp(torch.cat(feats, dim=-1))
        coord_term = rel * self.coord_mlp(m_ij)
        return m_ij, coord_term

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        m_ij, coord_term = inputs
        h_agg = scatter(m_ij, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        x_agg = scatter(coord_term, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
        return h_agg, x_agg

    def update(self, aggr_out):
        return aggr_out


class PocketConditionedEGNN(nn.Module):
    """Stack of EGNN layers operating on a combined pocket+ligand graph.

    Node features are built from atom-type / residue-type embeddings plus
    physicochemical descriptors (charge, hydrophobicity, H-bond donor/acceptor
    flags) for pocket atoms, and an optional diffusion-timestep embedding
    broadcast to every ligand node.
    """

    def __init__(self, cfg: EGNNConfig, physchem_dim: int = 6, time_embed: bool = True):
        super().__init__()
        self.cfg = cfg
        self.time_embed = time_embed
        self.atom_type_embed = nn.Embedding(cfg.n_atom_types, cfg.node_feature_dim)
        self.residue_type_embed = nn.Embedding(cfg.n_residue_types, cfg.node_feature_dim)
        self.is_ligand_embed = nn.Embedding(2, cfg.node_feature_dim)
        self.physchem_proj = nn.Linear(physchem_dim, cfg.node_feature_dim)

        in_dim = cfg.node_feature_dim
        self.input_proj = nn.Linear(in_dim, cfg.hidden_dim)
        if time_embed:
            self.time_proj = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )

        self.layers = nn.ModuleList(
            [EGNNLayer(cfg.hidden_dim, cfg.edge_feature_dim) for _ in range(cfg.n_layers)]
        )

    def build_node_features(
        self,
        atom_type_idx: Tensor,
        residue_type_idx: Tensor,
        is_ligand: Tensor,
        physchem: Tensor,
    ) -> Tensor:
        h = (
            self.atom_type_embed(atom_type_idx)
            + self.residue_type_embed(residue_type_idx)
            + self.is_ligand_embed(is_ligand.long())
            + self.physchem_proj(physchem)
        )
        return h

    def forward(
        self,
        atom_type_idx: Tensor,
        residue_type_idx: Tensor,
        is_ligand: Tensor,
        physchem: Tensor,
        coords: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        timestep: Tensor | None = None,
        batch: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """batch: (N,) graph-assignment index (as in torch_geometric.Batch),
        required together with `timestep` (one scalar per graph) so the
        per-graph diffusion-time embedding is broadcast onto the right nodes
        when several pocket+ligand graphs are batched together.
        """
        h = self.build_node_features(atom_type_idx, residue_type_idx, is_ligand, physchem)
        h = self.input_proj(h)

        if self.time_embed and timestep is not None:
            t_emb = self.time_proj(sinusoidal_time_embedding(timestep, self.cfg.hidden_dim))
            if batch is None:
                # Single graph: one timestep, broadcast to every node.
                h = h + t_emb.expand(h.shape[0], -1)
            else:
                h = h + t_emb[batch]

        x = coords
        update_mask = is_ligand.bool()
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr, update_mask=update_mask)
        return h, x
