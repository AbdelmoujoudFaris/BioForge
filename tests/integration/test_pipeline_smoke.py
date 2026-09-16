"""End-to-end smoke test: the full LangGraph pipeline against the bundled
1CRN structure, with tiny generation parameters so it runs in well under a
minute instead of the multi-minute runs default settings would take. Marked
`integration` since it makes real outbound network calls (RCSB fallbacks,
PubChem, RCSB anti-target fetch); skipped automatically when offline.
"""
from __future__ import annotations

import pytest

from bioforge.common.schemas import PipelineRunRequest, TargetIntake, TargetSource


@pytest.mark.integration
def test_full_pipeline_runs_end_to_end(network_available):
    if not network_available:
        pytest.skip("no network access in this environment")

    from bioforge.orchestrator.graph import run_pipeline

    req = PipelineRunRequest(
        target=TargetIntake(source=TargetSource.PDB_ID, value="1CRN", name="crambin"),
        n_candidates=2,
        max_atoms=8,
        n_timesteps=8,
        run_refinement=True,
        run_retrieval=True,
        run_mechanism=False,  # exercised separately in test_mechanism_integration (slow: many real network calls)
        seed=3,
    )
    result = run_pipeline(req)

    assert result.structure.structure_method == "experimental_pdb"
    assert len(result.pocket.candidates) >= 1
    assert result.generation.n_requested == 2
    assert len(result.reports) >= 1
    for report in result.reports:
        assert report.admet is not None
        assert report.synthesis is not None


@pytest.mark.integration
def test_mechanistic_hypothesis_end_to_end(network_available):
    if not network_available:
        pytest.skip("no network access in this environment")

    from bioforge.core.mechanism import generate_mechanistic_hypothesis

    result = generate_mechanistic_hypothesis("Compound X", "PTGS2", off_targets=["PTGS1"], indication="inflammation")
    assert result.causal_chain[0].actor == "Compound X"
    assert result.causal_chain[0].target == "PTGS2"
    assert 0.0 <= result.overall_confidence <= 1.0
    assert "PTGS2" in result.narrative
