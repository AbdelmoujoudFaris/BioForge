from bioforge.core.generation_ensemble import (
    DiffSBDDAdapter,
    RankedCandidate,
    generate_ensemble,
    multi_objective_rank,
)


def test_stub_adapters_raise_not_implemented():
    import pytest

    with pytest.raises(NotImplementedError):
        DiffSBDDAdapter().generate()


def test_generate_ensemble_produces_ranked_candidates():
    ranked = generate_ensemble(
        pdb_id="1CRN",
        generator_names=["equivariant_diffusion", "diffsbdd"],  # diffsbdd should be skipped, not crash the call
        n_candidates=2,
        max_atoms=8,
        n_timesteps=8,
        target_name="crambin",
        seed=7,
    )
    assert len(ranked) == 2
    assert all(c.generator_source == "equivariant_diffusion" for c in ranked)


def test_multi_objective_rank_flags_pareto_front():
    # binding_affinity_kcal_mol: lower (more negative) = tighter/better binding.
    candidates = [
        RankedCandidate("a", "CCO", True, "equivariant_diffusion", -8.0, 0.8, 2.0, 0.9),
        RankedCandidate("b", "CCC", True, "equivariant_diffusion", -3.0, 0.2, 8.0, 0.1),
        RankedCandidate("c", "CCN", True, "equivariant_diffusion", -8.0, 0.8, 2.0, 0.9),  # dominated-equal to a
    ]
    ranked = multi_objective_rank(candidates)
    assert ranked[0].pareto_optimal is True
    # candidate "b" is strictly worse on every objective than "a" -> not on the front
    assert ranked[1].pareto_optimal is False
