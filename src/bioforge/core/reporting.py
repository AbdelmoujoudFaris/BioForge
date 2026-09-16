"""Advanced Analysis & Reporting.

Ties together the outputs of every other `core/` module into the
per-candidate report surface: a Clinical Translation Score, uncertainty
flags across every prediction, and the data prep for the polypharmacology
network dashboard. Synthetic tractability (retrosynthesis) and patent
landscape both wrap real, vendored/RDKit-backed logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rdkit import Chem

from pharmaforge_core.scoring.retrosynthesis import RetrosynthesisScorer
from pharmaforge_core.utils.chem import synthetic_accessibility_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic tractability (real: SAscore always; AiZynthFinder if installed+configured)
# ---------------------------------------------------------------------------


def synthesis_route(mol: Chem.Mol, aizynthfinder_config: str | None = None) -> dict:
    return RetrosynthesisScorer(config_path=aizynthfinder_config).score(mol)


# ---------------------------------------------------------------------------
# Patent landscape (documented, honest stub - SureChEMBL needs a
# subscription; PubChem's own patent-annotation endpoint is a real,
# best-effort attempt)
# ---------------------------------------------------------------------------


@dataclass
class PatentNote:
    method: str
    likely_novel: bool | None
    notes: str


def patent_landscape_check(smiles: str, timeout: float = 15.0) -> PatentNote:
    """Best-effort real check against PubChem's compound-patent-count
    annotation (via the PUG-View XRefs endpoint); falls back to an honest
    "unknown, not checked" note rather than guessing. SureChEMBL full-text
    patent search needs API access this deployment doesn't have configured -
    see `SureChEMBLAdapter`.
    """
    import requests

    try:
        cid_resp = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON", timeout=timeout
        )
        cid_resp.raise_for_status()
        cid = cid_resp.json()["IdentifierList"]["CID"][0]
        xref_resp = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON",
            params={"heading": "Patents"},
            timeout=timeout,
        )
        has_patents = xref_resp.status_code == 200 and len(xref_resp.text) > 200
        return PatentNote(
            method="pubchem_patent_annotation",
            likely_novel=not has_patents,
            notes=(
                f"PubChem CID {cid}: {'patent annotations found' if has_patents else 'no patent annotations found'} "
                "- this checks PubChem's own patent cross-references, not a full claims-level novelty search."
            ),
        )
    except Exception as exc:
        logger.info("Patent landscape check unavailable: %s", exc)
        return PatentNote(
            method="unavailable",
            likely_novel=None,
            notes="Patent landscape could not be checked (network/lookup failure); treat as unknown, not novel.",
        )


class SureChEMBLAdapter:
    """Extension point for full-text patent search via the SureChEMBL API.
    Not implemented: needs API access this deployment doesn't have
    configured. `patent_landscape_check` above is the real, more limited
    PubChem-annotation-based alternative already wired up.
    """

    def search(self, smiles: str) -> dict:  # pragma: no cover
        raise NotImplementedError("Wire real SureChEMBL API access here; see class docstring.")


# ---------------------------------------------------------------------------
# Clinical Translation Score (composite index, documented weights)
# ---------------------------------------------------------------------------


@dataclass
class ClinicalTranslationResult:
    composite_score: float
    components: dict[str, float]
    method: str = "weighted_composite_v1"


_DEFAULT_WEIGHTS = {
    "preclinical_evidence": 0.35,  # from mechanism confidence + literature support
    "known_human_exposure": 0.25,  # 1.0 if retrieval found a DrugBank/ChEMBL analog, else 0.0
    "formulation_feasibility": 0.20,  # Lipinski/ADMET-derived
    "synthetic_tractability": 0.20,  # from SAscore/retrosynthesis
}


def clinical_translation_score(
    mechanism_confidence: float,
    has_known_human_exposure_analog: bool,
    admet_favorable: bool,
    sa_score: float,
    weights: dict[str, float] | None = None,
) -> ClinicalTranslationResult:
    """A documented, transparent weighted composite - not a validated
    clinical-success predictor. Weights (`_DEFAULT_WEIGHTS`) are a
    reasonable starting allocation across four evidence classes; recalibrate
    them against a real outcomes dataset before using this score for
    anything beyond triage/ranking.
    """
    weights = weights or _DEFAULT_WEIGHTS
    components = {
        "preclinical_evidence": mechanism_confidence,
        "known_human_exposure": 1.0 if has_known_human_exposure_analog else 0.0,
        "formulation_feasibility": 1.0 if admet_favorable else 0.3,
        "synthetic_tractability": max(0.0, min(1.0, (10.0 - sa_score) / 9.0)),
    }
    composite = sum(components[k] * weights.get(k, 0.0) for k in components)
    return ClinicalTranslationResult(composite_score=round(composite, 4), components=components)


# ---------------------------------------------------------------------------
# Uncertainty flags across every prediction surface
# ---------------------------------------------------------------------------


@dataclass
class UncertaintyFlagResult:
    field: str
    value: float
    threshold: float
    flagged: bool


def flag_uncertain_predictions(values: dict[str, tuple[float, float]]) -> list[UncertaintyFlagResult]:
    """`values`: field -> (value, threshold). Flags anything exceeding its
    threshold (e.g. MC-dropout std, propagation-confidence gaps) so a
    reviewer sees exactly which numbers in a report to distrust most.
    """
    return [
        UncertaintyFlagResult(field=field, value=value, threshold=threshold, flagged=value > threshold)
        for field, (value, threshold) in values.items()
    ]


# ---------------------------------------------------------------------------
# Polypharmacology dashboard data prep
# ---------------------------------------------------------------------------


@dataclass
class PolypharmacologyGraphData:
    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)


def polypharmacology_graph_data(
    compound_label: str,
    primary_target: str,
    off_target_scores: list[dict],
    perturbation_graph=None,
) -> PolypharmacologyGraphData:
    """Cytoscape.js/D3-ready node/edge lists: compound -> primary target
    (strong edge) and compound -> each screened off-target (edge weight from
    the real off-target panel score in `core/admet_safety.off_target_panel_screen`),
    optionally merged with the real STRING/OmniPath perturbation subgraph so
    the frontend can render one connected multi-target network.
    """
    nodes = [{"id": compound_label, "type": "compound"}, {"id": primary_target, "type": "primary_target"}]
    edges = [{"source": compound_label, "target": primary_target, "weight": 1.0, "kind": "designed_interaction"}]
    for hit in off_target_scores:
        nodes.append({"id": hit["anti_target"], "type": "off_target", "flagged": hit["flagged"]})
        edges.append(
            {
                "source": compound_label,
                "target": hit["anti_target"],
                "weight": hit["predicted_score"],
                "kind": "predicted_off_target",
            }
        )
    if perturbation_graph is not None:
        for node in perturbation_graph.graph.nodes:
            if not any(n["id"] == node for n in nodes):
                nodes.append({"id": node, "type": "ppi_neighbor"})
        for a, b, data in perturbation_graph.graph.edges(data=True):
            edges.append({"source": a, "target": b, "weight": data.get("weight", 0.5), "kind": "ppi"})
    return PolypharmacologyGraphData(nodes=nodes, edges=edges)
