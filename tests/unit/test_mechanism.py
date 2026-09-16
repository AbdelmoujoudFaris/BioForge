import networkx as nx

from bioforge.core.mechanism import PathwayHit, PerturbationGraph, build_causal_chain, propagate_perturbation


def _toy_graph() -> PerturbationGraph:
    g = nx.Graph()
    g.add_edge("TARGET", "NEIGHBOR1", weight=0.9)
    g.add_edge("TARGET", "NEIGHBOR2", weight=0.5)
    g.add_edge("NEIGHBOR1", "NEIGHBOR2", weight=0.3)
    return PerturbationGraph(graph=g, primary_target="TARGET", off_targets=[], source="synthetic_toy")


def test_propagate_perturbation_seeds_dominate():
    pg = _toy_graph()
    scores = propagate_perturbation(pg)
    assert scores["TARGET"] > 0
    assert set(scores) == {"TARGET", "NEIGHBOR1", "NEIGHBOR2"}


def test_propagate_perturbation_empty_graph_returns_empty_dict():
    pg = PerturbationGraph(graph=nx.Graph(), primary_target="TARGET", off_targets=[], source="empty")
    assert propagate_perturbation(pg) == {}


def test_build_causal_chain_starts_with_inhibition_step():
    hits = [PathwayHit(pathway_id="path:hsa00590", pathway_name="Arachidonic acid metabolism",
                        member_genes=["TARGET"], aggregate_perturbation_score=0.4)]
    chain = build_causal_chain("Compound X", "TARGET", {"TARGET": 0.4}, hits)
    assert chain[0].actor == "Compound X"
    assert chain[0].relation == "inhibits"
    assert chain[0].target == "TARGET"
    assert chain[1].target == "Arachidonic acid metabolism"
    assert 0.0 <= chain[1].confidence <= 1.0
