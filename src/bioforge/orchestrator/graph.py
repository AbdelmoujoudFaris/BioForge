"""Pipeline orchestration: a LangGraph `StateGraph` wiring every stage
(structure -> generate -> refine/retrieval/admet/mechanism -> report) into
one reproducible, resumable run.

Every node function below calls straight into the relevant `bioforge.core`
module *in-process*. That's a deliberate deployment choice, not a
architectural limitation: because every stage is also exposed as its own
FastAPI microservice (`bioforge.services.*`), a distributed deployment can
swap any node body for an HTTP call through `bioforge.sdk.client` (or a
Celery task via `bioforge.common.messaging.as_distributed_task`) to run that
stage on a separate GPU-scheduled worker pool, without changing the graph
topology. In-process calls are what the test suite and local `bioforge run`
CLI use, since spinning up eight servers isn't required to exercise the real
pipeline logic.
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from bioforge.common.schemas import (
    ADMETProfile,
    Candidate,
    CandidateReport,
    CausalChainStep,
    ClinicalTranslationScore,
    GeneratorSource,
    MechanisticHypothesis,
    ObjectiveScores,
    PatentLandscapeNote,
    PipelineRunRequest,
    PipelineRunResult,
    PocketCandidate,
    PocketResult,
    PoseUncertainty,
    RefinementResult,
    RetrievalHit,
    RetrievalResult,
    StructureRecord,
    SynthesisRoute,
    TargetIntake,
    UncertaintyFlag,
)
from bioforge.core import admet_safety, generation_ensemble, mechanism, refinement, reporting, retrieval, structure_intel
from pharmaforge_core.config import PocketConfig, ScoringConfig
from pharmaforge_core.data.pdb_download import fetch_structure
from pharmaforge_core.data.pocket_extraction import (
    extract_pocket_around_ligand,
    extract_pocket_around_point,
    geometric_center_fallback,
)
from pharmaforge_core.data.preprocessing import ligand_mol_to_tensors, pocket_to_tensors
from pharmaforge_core.models.scoring import ScoringStack
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

_TOP_K_FOR_DEEP_ANALYSIS = 3  # refine/retrieval/admet/mechanism only run on the top-K ranked candidates


class PipelineState(TypedDict, total=False):
    request: PipelineRunRequest
    run_id: str
    started_at: datetime
    stage_timings_ms: dict[str, float]
    structure: StructureRecord
    pocket: PocketResult
    candidates: list[Candidate]
    reports: list[CandidateReport]


def _timed(state: PipelineState, stage: str, fn):
    t0 = time.time()
    result = fn()
    state.setdefault("stage_timings_ms", {})[stage] = (time.time() - t0) * 1000
    return result


def node_structure(state: PipelineState) -> PipelineState:
    req = state["request"]

    def run():
        resolved = structure_intel.resolve_target(req.target.source.value, req.target.value, name=req.target.name)
        scored = structure_intel.detect_pockets(resolved, ligand_resname=req.target.ligand_resname)
        pocket_result = PocketResult(
            target_name=req.target.name,
            candidates=[
                PocketCandidate(
                    pocket_id=f"pocket_{i}",
                    center=tuple(float(v) for v in s.center),
                    n_atoms=len(s.pocket.elements),
                    classical_score=s.classical_score,
                    geometric_dl_score=s.geometric_dl_score,
                    combined_score=s.combined_score,
                    method=s.method,
                )
                for i, s in enumerate(scored)
            ],
            selected_pocket_id="pocket_0",
        )
        structure_record = StructureRecord(
            target_name=resolved.name,
            resolved_pdb_id=resolved.resolved_pdb_id,
            structure_method=resolved.structure_method,
            sequence=resolved.sequence,
            pdb_path=str(resolved.pdb_path),
            plddt_mean=resolved.plddt_mean,
            provenance=resolved.provenance,
        )
        return structure_record, pocket_result

    structure_record, pocket_result = _timed(state, "structure", run)
    state["structure"] = structure_record
    state["pocket"] = pocket_result
    return state


def node_generate(state: PipelineState) -> PipelineState:
    req = state["request"]

    def run():
        ranked = generation_ensemble.generate_ensemble(
            pdb_id=req.target.value,
            generator_names=[g.value for g in req.generators],
            n_candidates=req.n_candidates,
            max_atoms=req.max_atoms,
            n_timesteps=req.n_timesteps,
            ligand_resname=req.target.ligand_resname,
            anti_target_pdb_ids=req.anti_targets,
            target_name=req.target.name,
            seed=req.seed,
        )
        return [
            Candidate(
                candidate_id=c.candidate_id,
                smiles=c.smiles,
                valid=c.valid,
                generator_source=GeneratorSource(c.generator_source)
                if c.generator_source in GeneratorSource._value2member_map_
                else GeneratorSource.EQUIVARIANT_DIFFUSION,
                scores=ObjectiveScores(
                    binding_affinity_kcal_mol=c.binding_affinity_kcal_mol,
                    sa_score=c.sa_score,
                    qed=c.qed,
                    admet_desirability=c.admet_desirability,
                ),
                pareto_optimal=c.pareto_optimal,
            )
            for c in ranked
        ]

    state["candidates"] = _timed(state, "generate", run)
    return state


def _rank_for_deep_analysis(candidates: list[Candidate]) -> list[Candidate]:
    valid = [c for c in candidates if c.valid]
    valid.sort(key=lambda c: (not c.pareto_optimal, c.scores.binding_affinity_kcal_mol or 0.0))
    return valid[:_TOP_K_FOR_DEEP_ANALYSIS]


def node_deep_analysis(state: PipelineState) -> PipelineState:
    """Runs refinement, retrieval, ADMET, and mechanism for the top-K
    candidates only - real physics/network calls per candidate are too
    expensive to run against every generated molecule in a request handler;
    ranking (Pareto-optimal first) already happened in `node_generate`.
    """
    req = state["request"]
    top_candidates = _rank_for_deep_analysis(state.get("candidates", []))
    scoring_stack = ScoringStack(ScoringConfig())

    reports: list[CandidateReport] = []

    def run():
        for candidate in top_candidates:
            reports.append(_build_candidate_report(req, candidate, scoring_stack))
        return reports

    state["reports"] = _timed(state, "deep_analysis", run)
    return state


def _build_candidate_report(req: PipelineRunRequest, candidate: Candidate, scoring_stack: ScoringStack) -> CandidateReport:
    report = CandidateReport(candidate=candidate)
    if not candidate.smiles:
        return report

    mol = Chem.MolFromSmiles(candidate.smiles)
    if mol is None:
        return report

    if req.run_refinement:
        try:
            report.refinement = _run_refinement(req, candidate, scoring_stack, mol)
        except Exception as exc:
            logger.warning("Refinement failed for %s: %s", candidate.candidate_id, exc)

    if req.run_retrieval:
        try:
            retrieval_report = retrieval.retrieve_similar_compounds(candidate.smiles)
            report.retrieval = RetrievalResult(
                candidate_id=candidate.candidate_id,
                hits=[
                    RetrievalHit(
                        database=h.database, external_id=h.external_id, name=h.name, smiles=h.smiles,
                        tanimoto_ecfp4=h.tanimoto_ecfp4, embedding_similarity=h.embedding_similarity, method=h.method,
                    )
                    for h in retrieval_report.hits
                ],
            )
        except Exception as exc:
            logger.warning("Retrieval failed for %s: %s", candidate.candidate_id, exc)

    admet_profile = None
    try:
        result = admet_safety.profile_admet(mol)
        admet_profile = ADMETProfile(
            candidate_id=candidate.candidate_id,
            human_intestinal_absorption=result.human_intestinal_absorption,
            blood_brain_barrier_permeant=result.blood_brain_barrier_permeant,
            cyp_inhibition_risk=result.cyp_inhibition_risk,
            herg_liability_risk=result.herg_liability_risk,
            method=result.method,
        )
        report.admet = admet_profile
    except Exception as exc:
        logger.warning("ADMET profiling failed for %s: %s", candidate.candidate_id, exc)

    if req.run_mechanism:
        try:
            hyp = mechanism.generate_mechanistic_hypothesis(
                compound_label=f"Candidate {candidate.candidate_id[:8]}",
                primary_target=req.target.name,
                off_targets=list(req.anti_targets.keys()),
                indication=f"activity at {req.target.name}",
            )
            report.mechanism = MechanisticHypothesis(
                candidate_id=candidate.candidate_id,
                target_name=req.target.name,
                narrative=hyp.narrative,
                causal_chain=[
                    CausalChainStep(actor=s.actor, relation=s.relation, target=s.target, confidence=s.confidence,
                                     supporting_literature_count=s.supporting_literature_count)
                    for s in hyp.causal_chain
                ],
                predicted_moa=hyp.predicted_moa,
                resistance_mechanisms=hyp.resistance_mechanisms,
                biomarker_suggestions=hyp.biomarker_suggestions,
                overall_confidence=hyp.overall_confidence,
                method=hyp.method,
            )
        except Exception as exc:
            logger.warning("Mechanism analysis failed for %s: %s", candidate.candidate_id, exc)

    try:
        synth = reporting.synthesis_route(mol)
        report.synthesis = SynthesisRoute(
            method=synth["method"], route_found=synth["route_found"], n_steps=synth["n_steps"], sa_score=synth["sa_score"]
        )
    except Exception as exc:
        logger.warning("Synthesis scoring failed for %s: %s", candidate.candidate_id, exc)

    if report.synthesis and report.mechanism and report.admet:
        clinical = reporting.clinical_translation_score(
            mechanism_confidence=report.mechanism.overall_confidence,
            has_known_human_exposure_analog=bool(report.retrieval and any(h.tanimoto_ecfp4 and h.tanimoto_ecfp4 > 0.85 for h in report.retrieval.hits)),
            admet_favorable=bool(admet_profile and admet_profile.human_intestinal_absorption),
            sa_score=report.synthesis.sa_score,
        )
        report.clinical_translation = ClinicalTranslationScore(
            candidate_id=candidate.candidate_id, composite_score=clinical.composite_score, components=clinical.components,
            method=clinical.method,
        )

    if report.refinement and report.refinement.pose_uncertainty:
        report.uncertainty_flags = [
            UncertaintyFlag(
                field="pose_uncertainty_std", value=report.refinement.pose_uncertainty.std, threshold=0.75,
                flagged=report.refinement.pose_uncertainty.flagged_low_confidence,
            )
        ]

    return report


def _run_refinement(req: PipelineRunRequest, candidate: Candidate, scoring_stack: ScoringStack, mol: Chem.Mol) -> RefinementResult:
    mol_3d = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_3d, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol_3d)
    mol_3d = Chem.RemoveHs(mol_3d)

    equilibration = refinement.equilibrate_ligand(mol_3d, n_md_steps=100)

    pdb_path = fetch_structure(req.target.value)
    if req.target.ligand_resname:
        pocket = extract_pocket_around_ligand(pdb_path, req.target.ligand_resname, PocketConfig())
    else:
        center = geometric_center_fallback(pdb_path)
        pocket = extract_pocket_around_point(pdb_path, center, PocketConfig())
    pocket_tensors = pocket_to_tensors(pocket)
    ligand_tensors = ligand_mol_to_tensors(mol_3d)

    pose = refinement.refine_pose(
        scoring_stack, pocket_tensors, ligand_tensors["atom_type_idx"], ligand_tensors["physchem"], ligand_tensors["coords"],
    )
    import torch

    uncertainty = refinement.mc_dropout_uncertainty(
        scoring_stack, pocket_tensors, ligand_tensors["atom_type_idx"], ligand_tensors["physchem"],
        torch.as_tensor(pose.coords, dtype=torch.float32),
    )
    return RefinementResult(
        candidate_id=candidate.candidate_id,
        docking_score_kcal_mol=pose.final_score,
        pose_uncertainty=PoseUncertainty(
            method=uncertainty.method, mean=uncertainty.mean, std=uncertainty.std, n_samples=uncertainty.n_samples,
            flagged_low_confidence=uncertainty.flagged_low_confidence,
        ),
        md_equilibration={
            "method": equilibration.method, "rmsd_to_input_angstrom": equilibration.rmsd_to_input_angstrom,
            "converged": equilibration.converged,
        },
        method="openmm_generic_ff+differentiable_scoring_gradient_refinement",
    )


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("structure", node_structure)
    graph.add_node("generate", node_generate)
    graph.add_node("deep_analysis", node_deep_analysis)
    graph.set_entry_point("structure")
    graph.add_edge("structure", "generate")
    graph.add_edge("generate", "deep_analysis")
    graph.add_edge("deep_analysis", END)
    return graph.compile()


_COMPILED_GRAPH = None


def get_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_graph()
    return _COMPILED_GRAPH


def run_pipeline(request: PipelineRunRequest) -> PipelineRunResult:
    started_at = datetime.now(timezone.utc)
    initial_state: PipelineState = {"request": request, "run_id": str(uuid.uuid4()), "stage_timings_ms": {}}
    final_state = get_graph().invoke(initial_state)
    return PipelineRunResult(
        run_id=final_state["run_id"],
        target=request.target,
        structure=final_state["structure"],
        pocket=final_state["pocket"],
        generation=_generation_result(request, final_state.get("candidates", [])),
        reports=final_state.get("reports", []),
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
        stage_timings_ms=final_state.get("stage_timings_ms", {}),
    )


def _generation_result(request: PipelineRunRequest, candidates: list[Candidate]):
    from bioforge.common.schemas import GenerationResult

    return GenerationResult(
        target_name=request.target.name,
        candidates=candidates,
        n_requested=request.n_candidates,
        n_valid=sum(1 for c in candidates if c.valid),
    )
