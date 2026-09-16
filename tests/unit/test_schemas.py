from bioforge.common.schemas import (
    Candidate,
    GeneratorSource,
    ObjectiveScores,
    PipelineRunRequest,
    TargetIntake,
    TargetSource,
)


def test_target_intake_defaults():
    target = TargetIntake(source=TargetSource.PDB_ID, value="6LU7")
    assert target.name == "target"
    assert target.ligand_resname is None


def test_pipeline_run_request_defaults():
    req = PipelineRunRequest(target=TargetIntake(source=TargetSource.PDB_ID, value="6LU7"))
    assert req.n_candidates == 10
    assert req.generators == [GeneratorSource.EQUIVARIANT_DIFFUSION]
    assert req.run_mechanism is True


def test_candidate_round_trip():
    c = Candidate(
        candidate_id="abc123",
        smiles="CCO",
        valid=True,
        generator_source=GeneratorSource.EQUIVARIANT_DIFFUSION,
        scores=ObjectiveScores(binding_affinity_kcal_mol=-7.2, qed=0.6, sa_score=3.1),
    )
    payload = c.model_dump_json()
    restored = Candidate.model_validate_json(payload)
    assert restored.scores.binding_affinity_kcal_mol == -7.2
    assert restored.pareto_optimal is False
