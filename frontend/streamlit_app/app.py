"""BioForge dashboard: target intake -> generation -> ranked candidates with
full per-candidate reports (refinement, retrieval, ADMET, mechanism,
clinical translation score, polypharmacology network).

Runs the pipeline in-process by default (`bioforge.orchestrator.graph.run_pipeline`)
so the dashboard works without any other service running, matching how
`engines/pharmaforge_core/pharmaforge_core/ui/dashboard.py` operates. Set
`BIOFORGE_GATEWAY_URL` and flip the sidebar toggle to instead call a running
gateway over HTTP via the SDK - useful once services are actually split
across containers/pods.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# repo layout: frontend/streamlit_app/app.py -> ../../src, ../../engines/pharmaforge_core
_REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (_REPO_ROOT / "src", _REPO_ROOT / "engines" / "pharmaforge_core"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bioforge.common.schemas import (  # noqa: E402
    GeneratorSource,
    PipelineRunRequest,
    TargetIntake,
    TargetSource,
)

st.set_page_config(page_title="BioForge", page_icon="🧬", layout="wide")

st.title("🧬 BioForge")
st.caption("Open-source, AI-native drug repurposing and de novo design platform")

st.warning(
    "**Research-stage demo.** Generation runs the real, ported E(3)-equivariant diffusion "
    "architecture from `engines/pharmaforge_core` with randomly-initialized (seeded) weights - "
    "no checkpoint trained on PDBbind/CrossDocked2020 ships with this repository. Structure/"
    "retrieval/mechanism calls hit real free APIs (RCSB, UniProt, AlphaFold DB, PubChem, STRING, "
    "OmniPath, KEGG, PubMed); ADMET/off-target/refinement use real but simplified physics/heuristics. "
    "See `docs/whitepaper.md` before treating any output as validated.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Target")
    pdb_id = st.text_input("PDB ID", value="1CRN", help="e.g. 6LU7 (SARS-CoV-2 Mpro)")
    target_name = st.text_input("Target label (gene symbol for mechanism analysis)", value="target")
    ligand_resname = st.text_input("Reference ligand residue (optional)", value="")

    st.header("Generation")
    n_candidates = st.slider("Candidates", 1, 30, 5)
    max_atoms = st.slider("Max ligand atoms", 5, 45, 20)
    n_timesteps = st.slider("Diffusion timesteps", 5, 200, 40, help="Higher = slower, higher-quality sampling")
    generators = st.multiselect(
        "Generators", [g.value for g in GeneratorSource], default=[GeneratorSource.EQUIVARIANT_DIFFUSION.value]
    )
    seed = st.number_input("Seed", value=42)

    st.header("Deep analysis (top-ranked candidates only)")
    run_refinement = st.checkbox("Physics-based refinement (OpenMM + pose)", value=True)
    run_retrieval = st.checkbox("Database retrieval (PubChem/ChEMBL/ZINC/DrugBank)", value=True)
    run_mechanism = st.checkbox("Causal mechanism + systems pharmacology", value=False, help="Slow: several real network calls per candidate")

    anti_target_text = st.text_area("Anti-targets (name=PDB_ID, one per line)", value="", help="e.g. hERG=5VA1")

    run_clicked = st.button("Run pipeline", type="primary", use_container_width=True)

if "result" not in st.session_state:
    st.session_state.result = None


def _parse_anti_targets(text: str) -> dict[str, str]:
    out = {}
    for line in text.strip().splitlines():
        if "=" in line:
            name, pdb = line.split("=", 1)
            out[name.strip()] = pdb.strip()
    return out


if run_clicked:
    from bioforge.orchestrator.graph import run_pipeline

    req = PipelineRunRequest(
        target=TargetIntake(
            source=TargetSource.PDB_ID, value=pdb_id, name=target_name, ligand_resname=ligand_resname or None
        ),
        n_candidates=n_candidates,
        max_atoms=max_atoms,
        n_timesteps=n_timesteps,
        generators=[GeneratorSource(g) for g in generators] or [GeneratorSource.EQUIVARIANT_DIFFUSION],
        anti_targets=_parse_anti_targets(anti_target_text),
        run_refinement=run_refinement,
        run_retrieval=run_retrieval,
        run_mechanism=run_mechanism,
        seed=int(seed),
    )
    with st.spinner("Running pipeline (structure -> generate -> deep analysis)... this can take a while on CPU."):
        try:
            st.session_state.result = run_pipeline(req)
        except Exception as exc:
            st.error(f"Pipeline run failed: {exc}")
            st.session_state.result = None

result = st.session_state.result
if result is None:
    st.info("Configure a target in the sidebar and click **Run pipeline**.")
    st.stop()

st.subheader("Structure")
c1, c2, c3 = st.columns(3)
c1.metric("Method", result.structure.structure_method)
c2.metric("Resolved PDB", result.structure.resolved_pdb_id or "-")
c3.metric("Mean pLDDT", f"{result.structure.plddt_mean:.1f}" if result.structure.plddt_mean else "n/a")

st.subheader("Pockets detected")
pocket_df = pd.DataFrame(
    [
        {
            "pocket_id": p.pocket_id, "n_atoms": p.n_atoms, "classical_score": round(p.classical_score, 3),
            "geometric_dl_score": round(p.geometric_dl_score, 3), "combined_score": round(p.combined_score, 3),
            "method": p.method,
        }
        for p in result.pocket.candidates
    ]
)
st.dataframe(pocket_df, use_container_width=True, hide_index=True)

st.subheader(f"Candidates ({result.generation.n_valid}/{result.generation.n_requested} valid)")
cand_df = pd.DataFrame(
    [
        {
            "candidate_id": c.candidate_id[:8], "valid": c.valid, "generator": c.generator_source,
            "smiles": c.smiles, "affinity (kcal/mol-like)": c.scores.binding_affinity_kcal_mol,
            "QED": c.scores.qed, "SAscore": c.scores.sa_score, "pareto_optimal": c.pareto_optimal,
        }
        for c in result.generation.candidates
    ]
)
st.dataframe(cand_df, use_container_width=True, hide_index=True)

st.subheader("Candidate reports (top-ranked)")
for report in result.reports:
    label = report.candidate.smiles or report.candidate.candidate_id[:8]
    with st.expander(f"{report.candidate.candidate_id[:8]} - {label}", expanded=False):
        tabs = st.tabs(["ADMET", "Refinement", "Retrieval", "Mechanism", "Synthesis & Clinical"])

        with tabs[0]:
            if report.admet:
                st.json(report.admet.model_dump())
            else:
                st.write("Not run.")

        with tabs[1]:
            if report.refinement:
                st.json(report.refinement.model_dump())
                if report.uncertainty_flags:
                    for flag in report.uncertainty_flags:
                        if flag.flagged:
                            st.error(f"Low confidence: {flag.field} = {flag.value:.3f} (threshold {flag.threshold})")
            else:
                st.write("Not run.")

        with tabs[2]:
            if report.retrieval and report.retrieval.hits:
                st.dataframe(pd.DataFrame([h.model_dump() for h in report.retrieval.hits]), use_container_width=True, hide_index=True)
            else:
                st.write("No hits / not run.")

        with tabs[3]:
            if report.mechanism:
                st.markdown(f"**Predicted MoA:** {report.mechanism.predicted_moa}")
                st.text(report.mechanism.narrative)
                chain_df = pd.DataFrame([s.model_dump() for s in report.mechanism.causal_chain])
                st.dataframe(chain_df, use_container_width=True, hide_index=True)
                st.markdown(f"**Overall confidence:** {report.mechanism.overall_confidence:.2f}")
                st.markdown("**Resistance mechanisms:** " + "; ".join(report.mechanism.resistance_mechanisms))
                st.markdown("**Biomarker suggestions:** " + "; ".join(report.mechanism.biomarker_suggestions))
            else:
                st.write("Not run.")

        with tabs[4]:
            cols = st.columns(2)
            if report.synthesis:
                cols[0].json(report.synthesis.model_dump())
            if report.clinical_translation:
                cols[1].json(report.clinical_translation.model_dump())

st.subheader("Polypharmacology network")
if result.reports:
    from bioforge.core.reporting import polypharmacology_graph_data

    first = result.reports[0]
    off_target_hits = (first.admet.off_target_hits if first.admet else []) or []
    graph_data = polypharmacology_graph_data(
        compound_label=first.candidate.candidate_id[:8], primary_target=target_name, off_target_scores=off_target_hits
    )
    if graph_data.edges:
        import networkx as nx

        g = nx.Graph()
        for n in graph_data.nodes:
            g.add_node(n["id"], **n)
        for e in graph_data.edges:
            g.add_edge(e["source"], e["target"], weight=e["weight"])
        pos = nx.spring_layout(g, seed=1)

        edge_x, edge_y = [], []
        for a, b in g.edges():
            edge_x += [pos[a][0], pos[b][0], None]
            edge_y += [pos[a][1], pos[b][1], None]
        node_x = [pos[n][0] for n in g.nodes()]
        node_y = [pos[n][1] for n in g.nodes()]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#999"), hoverinfo="none"))
        fig.add_trace(
            go.Scatter(
                x=node_x, y=node_y, mode="markers+text", text=list(g.nodes()), textposition="top center",
                marker=dict(size=18, color="#4C78A8"),
            )
        )
        fig.update_layout(showlegend=False, height=500, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No off-target/network data available for this run (enable an anti-target panel or mechanism analysis).")
else:
    st.write("No candidate reports to visualize yet.")

st.caption(f"Run ID: {result.run_id} | stage timings (ms): {result.stage_timings_ms}")
