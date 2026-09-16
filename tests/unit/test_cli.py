import sys

import pytest

from bioforge import cli


def test_run_command_invokes_pipeline_and_prints_json(monkeypatch, capsys):
    from bioforge.common.schemas import (
        GenerationResult,
        PipelineRunResult,
        PocketResult,
        StructureRecord,
        TargetIntake,
        TargetSource,
    )
    from datetime import datetime, timezone

    fake_result = PipelineRunResult(
        run_id="r1",
        target=TargetIntake(source=TargetSource.PDB_ID, value="1CRN"),
        structure=StructureRecord(target_name="target", structure_method="experimental_pdb"),
        pocket=PocketResult(target_name="target", candidates=[], selected_pocket_id="pocket_0"),
        generation=GenerationResult(target_name="target", candidates=[], n_requested=1, n_valid=0),
        reports=[],
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )
    monkeypatch.setattr("bioforge.orchestrator.graph.run_pipeline", lambda req: fake_result)
    monkeypatch.setattr(sys, "argv", ["bioforge", "run", "--pdb-id", "1CRN", "--n-candidates", "1"])

    cli.main()

    out = capsys.readouterr().out
    assert '"run_id": "r1"' in out


def test_serve_command_rejects_unknown_service(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["bioforge", "serve", "not-a-real-service"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "Unknown service" in capsys.readouterr().err
