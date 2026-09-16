"""Pydantic schemas shared by every BioForge service.

This module is the contract every microservice speaks: the gateway, the
orchestrator graph, and the Python SDK all import from here rather than
redefining shapes locally. Keeping it dependency-free (no torch/rdkit
imports) means any service can import it without pulling in the heavy
science stack just to validate a request body.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Intake & structure intelligence
# ---------------------------------------------------------------------------


class TargetSource(str, Enum):
    PDB_ID = "pdb_id"
    UNIPROT_ID = "uniprot_id"
    ALPHAFOLD_ID = "alphafold_id"
    SEQUENCE = "sequence"


class TargetIntake(BaseModel):
    source: TargetSource
    value: str = Field(..., description="PDB ID, UniProt accession, AlphaFold DB ID, or raw sequence")
    ligand_resname: str | None = Field(None, description="Reference co-crystallized ligand 3-letter code")
    name: str = Field("target", description="Human-readable label used in reports")


class StructureRecord(BaseModel):
    target_name: str
    resolved_pdb_id: str | None = None
    structure_method: str  # "experimental_pdb" | "alphafold_db" | "boltz1_predicted" | "chai1_predicted"
    sequence: str | None = None
    pdb_path: str | None = None
    plddt_mean: float | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class PocketCandidate(BaseModel):
    pocket_id: str
    center: tuple[float, float, float]
    n_atoms: int
    classical_score: float = Field(..., description="fpocket/geometric-heuristic druggability score, 0-1")
    geometric_dl_score: float = Field(..., description="learned pocket-druggability GNN score, 0-1")
    combined_score: float
    method: str


class PocketResult(BaseModel):
    target_name: str
    candidates: list[PocketCandidate]
    selected_pocket_id: str


# ---------------------------------------------------------------------------
# Generative chemistry
# ---------------------------------------------------------------------------


class GeneratorSource(str, Enum):
    EQUIVARIANT_DIFFUSION = "equivariant_diffusion"  # real (ported PharmaForge EGNN diffusion)
    DIFFSBDD = "diffsbdd"  # stub adapter, documented extension point
    DIFFDOCK_L = "diffdock_l"  # stub adapter
    LATENT_DIFFUSION = "latent_diffusion"  # stub adapter


class GenerationRequest(BaseModel):
    target: TargetIntake
    pocket_id: str | None = None
    n_candidates: int = 20
    max_atoms: int = 45
    n_timesteps: int = Field(
        50,
        description=(
            "Diffusion sampling steps. The vendored engine defaults to 1000 for GPU "
            "deployments; the API default trades sample quality for CPU-interactive "
            "latency. Raise this for a GPU worker pool."
        ),
    )
    generators: list[GeneratorSource] = Field(default_factory=lambda: [GeneratorSource.EQUIVARIANT_DIFFUSION])
    anti_targets: dict[str, str] = Field(default_factory=dict, description="name -> PDB ID panel for selectivity")
    seed: int | None = 42


class ObjectiveScores(BaseModel):
    binding_affinity_kcal_mol: float | None = None
    selectivity_index: dict[str, float] = Field(default_factory=dict)
    sa_score: float | None = None
    qed: float | None = None
    admet_desirability: float | None = None


class Candidate(BaseModel):
    candidate_id: str
    smiles: str | None = None
    valid: bool
    generator_source: GeneratorSource
    scores: ObjectiveScores
    pareto_optimal: bool = False
    provenance: dict[str, Any] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    target_name: str
    candidates: list[Candidate]
    n_requested: int
    n_valid: int


# ---------------------------------------------------------------------------
# Physics-based refinement
# ---------------------------------------------------------------------------


class PoseUncertainty(BaseModel):
    method: str  # "mc_dropout_ensemble_disagreement"
    mean: float
    std: float
    n_samples: int
    flagged_low_confidence: bool


class RefinementResult(BaseModel):
    candidate_id: str
    docking_score_kcal_mol: float | None
    pose_uncertainty: PoseUncertainty | None
    md_equilibration: dict[str, Any] = Field(default_factory=dict)
    fep_estimate: dict[str, Any] | None = None
    method: str


# ---------------------------------------------------------------------------
# Retrieval & similarity
# ---------------------------------------------------------------------------


class RetrievalHit(BaseModel):
    database: str  # "pubchem" | "chembl" | "zinc" | "drugbank"
    external_id: str
    name: str | None
    smiles: str | None
    tanimoto_ecfp4: float | None = None
    embedding_similarity: float | None = None
    method: str


class RetrievalResult(BaseModel):
    candidate_id: str
    hits: list[RetrievalHit]


# ---------------------------------------------------------------------------
# ADMET & safety
# ---------------------------------------------------------------------------


class ADMETProfile(BaseModel):
    candidate_id: str
    human_intestinal_absorption: bool | None = None
    blood_brain_barrier_permeant: bool | None = None
    cyp_inhibition_risk: str | None = None
    herg_liability_risk: str | None = None
    off_target_hits: list[dict[str, Any]] = Field(default_factory=list)
    method: str


# ---------------------------------------------------------------------------
# Causal mechanistic interpretability & systems pharmacology (killer feature)
# ---------------------------------------------------------------------------


class PerturbationNode(BaseModel):
    protein: str
    role: str  # "primary_target" | "predicted_off_target"
    binding_confidence: float


class CausalChainStep(BaseModel):
    actor: str
    relation: str  # "inhibits" | "activates" | "downregulates" | "upregulates" | "suppresses"
    target: str
    confidence: float
    supporting_literature_count: int = 0


class MechanisticHypothesis(BaseModel):
    candidate_id: str
    target_name: str
    narrative: str
    causal_chain: list[CausalChainStep]
    predicted_moa: str
    resistance_mechanisms: list[str] = Field(default_factory=list)
    biomarker_suggestions: list[str] = Field(default_factory=list)
    overall_confidence: float
    method: str
    generated_at: datetime = Field(default_factory=_utcnow)


class PerturbationGraphSummary(BaseModel):
    candidate_id: str
    n_nodes: int
    n_edges: int
    primary_target: str
    off_targets: list[str]
    source: str  # "string" | "omnipath" | "bundled_sample"


# ---------------------------------------------------------------------------
# Advanced analysis & reporting
# ---------------------------------------------------------------------------


class SynthesisRoute(BaseModel):
    method: str
    route_found: bool | None
    n_steps: int | None
    sa_score: float


class PatentLandscapeNote(BaseModel):
    method: str
    likely_novel: bool | None
    notes: str


class ClinicalTranslationScore(BaseModel):
    candidate_id: str
    composite_score: float
    components: dict[str, float]
    method: str


class UncertaintyFlag(BaseModel):
    field: str
    value: float
    threshold: float
    flagged: bool


class CandidateReport(BaseModel):
    candidate: Candidate
    admet: ADMETProfile | None = None
    refinement: RefinementResult | None = None
    retrieval: RetrievalResult | None = None
    mechanism: MechanisticHypothesis | None = None
    perturbation_graph: PerturbationGraphSummary | None = None
    synthesis: SynthesisRoute | None = None
    patent: PatentLandscapeNote | None = None
    clinical_translation: ClinicalTranslationScore | None = None
    uncertainty_flags: list[UncertaintyFlag] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class PipelineRunRequest(BaseModel):
    target: TargetIntake
    n_candidates: int = 10
    max_atoms: int = 45
    n_timesteps: int = 50
    anti_targets: dict[str, str] = Field(default_factory=dict)
    generators: list[GeneratorSource] = Field(default_factory=lambda: [GeneratorSource.EQUIVARIANT_DIFFUSION])
    run_refinement: bool = True
    run_retrieval: bool = True
    run_mechanism: bool = True
    seed: int | None = 42


class PipelineRunResult(BaseModel):
    run_id: str
    target: TargetIntake
    structure: StructureRecord
    pocket: PocketResult
    generation: GenerationResult
    reports: list[CandidateReport]
    started_at: datetime
    finished_at: datetime
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
