"""Central configuration objects for PharmaForge.

All dataclasses here are plain, serializable configs consumed by the data
pipeline, models, generation loop and analysis suite. Keeping them in one
module means a training run, a checkpoint and a generation call all agree
on the same field names without importing each other's internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data_cache"

# Atom vocabulary used across the pocket graph, the ligand diffusion model
# and the scoring stack. Index 0 is reserved for padding.
ATOM_VOCAB = ["PAD", "C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "H", "OTHER"]
RESIDUE_VOCAB = [
    "PAD", "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR",
    "VAL", "OTHER",
]
BOND_TYPES = ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]


@dataclass
class PocketConfig:
    """How a binding pocket is carved out of a receptor structure."""

    radius_around_ligand: float = 8.0  # angstrom, used when a reference ligand exists
    radius_around_centroid: float = 10.0  # angstrom, used for blind pocket-centroid mode
    max_pocket_atoms: int = 600
    include_hydrogens: bool = False
    fpocket_min_druggability: float = 0.2


@dataclass
class EGNNConfig:
    node_feature_dim: int = 64
    hidden_dim: int = 128
    n_layers: int = 6
    edge_feature_dim: int = 16
    n_atom_types: int = len(ATOM_VOCAB)
    n_residue_types: int = len(RESIDUE_VOCAB)
    cutoff_radius: float = 10.0
    max_neighbors: int = 32


@dataclass
class DiffusionConfig:
    n_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    schedule: str = "cosine"  # "linear" | "cosine"
    max_ligand_atoms: int = 45
    egnn: EGNNConfig = field(default_factory=EGNNConfig)
    guidance_scale: float = 1.0  # classifier-free / scoring-guidance strength


@dataclass
class ActorCriticConfig:
    egnn: EGNNConfig = field(default_factory=EGNNConfig)
    n_transformer_layers: int = 4
    n_heads: int = 8
    action_space_size: int = len(ATOM_VOCAB) * len(BOND_TYPES)
    gamma: float = 0.99
    entropy_coef: float = 0.01


@dataclass
class ScoringConfig:
    egnn: EGNNConfig = field(default_factory=EGNNConfig)
    affinity_hidden_dim: int = 128
    fingerprint_dim: int = 6  # h-bond donor/acceptor, hydrophobic, pi-stack, halogen, salt-bridge
    dropout: float = 0.1


@dataclass
class SelectivityConfig:
    """Multi-objective selectivity-aware generation settings."""

    method: str = "nsga3"  # "nsga3" | "moead" | "penalty"
    population_size: int = 64
    n_generations: int = 40
    anti_target_penalty_weight: float = 1.0
    selectivity_index_floor: float = 1.0  # SI = Kd_offtarget / Kd_target, want >> 1


@dataclass
class GenerationConfig:
    mode: str = "de_novo"  # "de_novo" | "fragment_growing" | "scaffold_hopping"
    n_candidates: int = 100
    max_atoms: int = 45
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    selectivity: SelectivityConfig = field(default_factory=SelectivityConfig)
    pocket: PocketConfig = field(default_factory=PocketConfig)
    seed: int | None = 42
