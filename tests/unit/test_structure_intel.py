from bioforge.core.structure_intel import (
    detect_pockets,
    ligsite_grid_pockets,
    resolve_target,
    score_pocket_classical,
    score_pocket_geometric_dl,
)
from pharmaforge_core.data.pocket_extraction import extract_pocket_around_point, geometric_center_fallback


def test_resolve_target_pdb_id_uses_bundled_cache():
    resolved = resolve_target("pdb_id", "1CRN", name="crambin")
    assert resolved.structure_method == "experimental_pdb"
    assert resolved.pdb_path.exists()
    assert resolved.resolved_pdb_id == "1CRN"


def test_resolve_target_sequence_not_implemented():
    import pytest

    with pytest.raises(NotImplementedError):
        resolve_target("sequence", "MKT...", name="x")


def test_ligsite_grid_returns_at_least_one_point(bundled_pdb_path):
    points = ligsite_grid_pockets(bundled_pdb_path, search_radius=10.0, grid_spacing=2.5, max_candidates=3)
    assert len(points) >= 1
    assert points[0].shape == (3,)


def test_pocket_scoring_in_unit_range(bundled_pdb_path):
    center = geometric_center_fallback(bundled_pdb_path)
    pocket = extract_pocket_around_point(bundled_pdb_path, center)
    assert 0.0 <= score_pocket_classical(pocket) <= 1.0
    assert 0.0 <= score_pocket_geometric_dl(pocket) <= 1.0


def test_detect_pockets_ensemble_ranks_candidates():
    resolved = resolve_target("pdb_id", "1CRN", name="crambin")
    scored = detect_pockets(resolved, max_candidates=2)
    assert len(scored) >= 1
    scores = [s.combined_score for s in scored]
    assert scores == sorted(scores, reverse=True)
