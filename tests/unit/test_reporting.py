from bioforge.core.reporting import clinical_translation_score, flag_uncertain_predictions, polypharmacology_graph_data


def test_clinical_translation_score_components_sum_to_composite():
    result = clinical_translation_score(
        mechanism_confidence=0.8, has_known_human_exposure_analog=True, admet_favorable=True, sa_score=3.0,
    )
    assert 0.0 <= result.composite_score <= 1.0
    assert set(result.components) == {
        "preclinical_evidence", "known_human_exposure", "formulation_feasibility", "synthetic_tractability",
    }


def test_flag_uncertain_predictions():
    flags = flag_uncertain_predictions({"pose_std": (0.9, 0.75), "affinity_std": (0.1, 0.75)})
    flagged = {f.field: f.flagged for f in flags}
    assert flagged == {"pose_std": True, "affinity_std": False}


def test_polypharmacology_graph_data_includes_compound_and_targets():
    data = polypharmacology_graph_data(
        compound_label="X", primary_target="TARGET",
        off_target_scores=[{"anti_target": "hERG", "predicted_score": 0.4, "flagged": True}],
    )
    node_ids = {n["id"] for n in data.nodes}
    assert {"X", "TARGET", "hERG"} <= node_ids
    assert len(data.edges) == 2
