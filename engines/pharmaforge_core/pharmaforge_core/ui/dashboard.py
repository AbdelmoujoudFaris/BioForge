"""PharmaForge interactive dashboard.

Run with:

    streamlit run pharmaforge_core/ui/dashboard.py

Lets a user type a PDB ID, resolve/inspect the binding pocket, run
pocket-conditioned generation (de novo or fragment-growing), browse
candidates ranked by score or by multi-objective Pareto front (when
anti-targets are configured for selectivity-aware design), inspect a 3D
scatter of each candidate's atoms in the pocket, and download a
publication-ready HTML report.

No checkpoint is bundled with this repository (see docs/whitepaper.md for
why: a real PDBbind/CrossDocked2020 training run needs GPU time this
environment doesn't have). Without a checkpoint path, the dashboard runs the
exact same code path against randomly-initialized weights, in a clearly
labeled "demo / untrained weights" mode, with a small step count so it stays
interactive - this proves the pipeline end-to-end without pretending the
generated molecules are optimized.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from pharmaforge_core.analysis.diversity import internal_diversity, scaffold_diversity, uniqueness
from pharmaforge_core.analysis.physchem import profile_candidates, summarize
from pharmaforge_core.analysis.report import generate_report
from pharmaforge_core.analysis.selectivity_report import (
    affinity_heatmap,
    offtarget_risk_table,
    pareto_front_scatter,
)
from pharmaforge_core.config import (
    ATOM_VOCAB,
    DiffusionConfig,
    EGNNConfig,
    GenerationConfig,
    ScoringConfig,
    SelectivityConfig,
)
from pharmaforge_core.data.pdb_download import fetch_structure
from pharmaforge_core.data.pocket_extraction import (
    extract_pocket_around_ligand,
    extract_pocket_around_point,
    find_fpocket_cavities,
    geometric_center_fallback,
)
from pharmaforge_core.data.preprocessing import pocket_to_tensors
from pharmaforge_core.generation.sampler import GuidedDiffusionSampler
from pharmaforge_core.models.diffusion import LigandDiffusionModel
from pharmaforge_core.models.scoring import MultiTargetScoringEnsemble, ScoringStack
from pharmaforge_core.models.selectivity import SelectivityProfiler, pareto_front_indices

st.set_page_config(page_title="PharmaForge", page_icon=":material/science:", layout="wide")


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_models(hidden_dim: int, n_layers: int, n_timesteps: int, checkpoint_dir: str | None):
    egnn_cfg = EGNNConfig(node_feature_dim=hidden_dim, hidden_dim=hidden_dim, n_layers=n_layers, edge_feature_dim=0)
    diffusion_cfg = DiffusionConfig(n_timesteps=n_timesteps, egnn=egnn_cfg)
    scoring_cfg = ScoringConfig(egnn=egnn_cfg)

    diffusion_model = LigandDiffusionModel(diffusion_cfg)
    scoring_stack = ScoringStack(scoring_cfg)

    loaded_checkpoint = False
    if checkpoint_dir:
        diff_ckpt = Path(checkpoint_dir) / "diffusion.pt"
        score_ckpt = Path(checkpoint_dir) / "scoring.pt"
        import torch

        if diff_ckpt.exists():
            diffusion_model.load_state_dict(torch.load(diff_ckpt, map_location="cpu"))
            loaded_checkpoint = True
        if score_ckpt.exists():
            scoring_stack.load_state_dict(torch.load(score_ckpt, map_location="cpu"))
            loaded_checkpoint = True

    diffusion_model.eval()
    scoring_stack.eval()
    return diffusion_model, scoring_stack, loaded_checkpoint


@st.cache_data(show_spinner=False, ttl="1h")
def resolve_pocket_cached(pdb_id: str, ligand_resname: str | None):
    pdb_path = fetch_structure(pdb_id)
    if ligand_resname:
        pocket = extract_pocket_around_ligand(pdb_path, ligand_resname)
    else:
        centers = find_fpocket_cavities(pdb_path)
        center = centers[0] if centers else geometric_center_fallback(pdb_path)
        pocket = extract_pocket_around_point(pdb_path, center)
    return pocket, str(pdb_path)


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Target configuration")
    pdb_id = st.text_input("Primary target PDB ID", value="6LU7", help="e.g. 6LU7 (SARS-CoV-2 main protease)")
    ligand_resname = st.text_input(
        "Reference ligand residue name (optional)",
        value="",
        help="3-letter HETATM code to carve the pocket around a known ligand; leave blank for blind fpocket detection",
    )

    st.divider()
    st.subheader("Selectivity profiler (optional)")
    anti_target_raw = st.text_area(
        "Anti-target PDB IDs (one per line)",
        value="",
        help="Off-target proteins to penalize during generation, e.g. hERG (5VA1)",
    )
    anti_target_ids = [line.strip().upper() for line in anti_target_raw.splitlines() if line.strip()]

    st.divider()
    st.subheader("Generation settings")
    mode = st.selectbox("Mode", ["de_novo", "fragment_growing"], help="fragment_growing requires a seed SMILES below")
    seed_smiles = st.text_input("Seed SMILES (fragment-growing only)", value="") if mode == "fragment_growing" else None
    n_candidates = st.slider("Number of candidates", 1, 50, 8)
    max_atoms = st.slider("Max ligand heavy atoms", 5, 45, 20)

    with st.expander("Demo-mode model settings"):
        st.caption(
            "No pretrained checkpoint ships with this repo (see docs/whitepaper.md). "
            "These sliders control an untrained model's size/step-count so the dashboard "
            "stays responsive; point 'checkpoint directory' at real weights once trained."
        )
        hidden_dim = st.select_slider("Backbone hidden dim", [16, 32, 64, 128], value=32)
        n_layers = st.slider("EGNN layers", 1, 8, 3)
        n_timesteps = st.slider("Diffusion steps", 10, 200, 50)
        checkpoint_dir = st.text_input("Checkpoint directory", value="checkpoints")

    run_clicked = st.button("Run generation", type="primary", width="stretch")


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------

st.title("PharmaForge")
st.caption("Structure-aware de novo drug design: 3D diffusion generation, differentiable scoring, and multi-objective selectivity.")

if "candidates_df" not in st.session_state:
    st.session_state.candidates_df = None
    st.session_state.mols = []
    st.session_state.pocket = None
    st.session_state.pocket_pdb_path = None

pocket_col, info_col = st.columns([2, 1])

if pdb_id:
    try:
        with st.spinner(f"Resolving pocket for {pdb_id}..."):
            pocket, pocket_pdb_path = resolve_pocket_cached(pdb_id, ligand_resname or None)
        st.session_state.pocket = pocket
        st.session_state.pocket_pdb_path = pocket_pdb_path
    except Exception as exc:
        st.error(f"Could not resolve pocket for '{pdb_id}': {exc}")
        st.session_state.pocket = None

if st.session_state.pocket is not None:
    pocket = st.session_state.pocket
    with pocket_col, st.container(border=True):
        st.subheader(f"Pocket - {pdb_id}")
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=pocket.coords[:, 0], y=pocket.coords[:, 1], z=pocket.coords[:, 2],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=["orange" if h else "steelblue" for h in pocket.is_hydrophobic],
                    ),
                    text=[f"{r} ({e})" for r, e in zip(pocket.residue_names, pocket.elements)],
                    name="pocket atoms",
                )
            ]
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=420, scene=dict(aspectmode="data"))
        st.plotly_chart(fig, width="stretch")
    with info_col, st.container(border=True):
        st.metric("Pocket atoms", len(pocket.elements), border=False)
        st.metric("Hydrophobic atoms", int(pocket.is_hydrophobic.sum()), border=False)
        st.metric("H-bond donors", int(pocket.is_hbond_donor.sum()), border=False)
        st.metric("H-bond acceptors", int(pocket.is_hbond_acceptor.sum()), border=False)

if run_clicked and st.session_state.pocket is not None:
    diffusion_model, scoring_stack, loaded_checkpoint = load_models(hidden_dim, n_layers, n_timesteps, checkpoint_dir)
    if not loaded_checkpoint:
        st.warning(
            "Running with randomly-initialized (untrained) weights - no checkpoint found at "
            f"'{checkpoint_dir}'. Generated molecules are a plumbing demo, not optimized candidates. "
            "See docs/whitepaper.md for how to train real checkpoints."
        )

    pocket_tensors = pocket_to_tensors(st.session_state.pocket)

    selectivity_profiler = None
    anti_target_pockets = None
    if anti_target_ids:
        with st.spinner("Resolving anti-target pockets..."):
            anti_target_pockets = {}
            for anti_id in anti_target_ids:
                anti_pocket, _ = resolve_pocket_cached(anti_id, None)
                anti_target_pockets[anti_id] = pocket_to_tensors(anti_pocket)
        ensemble = MultiTargetScoringEnsemble(scoring_stack)
        selectivity_profiler = SelectivityProfiler(ensemble, SelectivityConfig())

    gen_cfg = GenerationConfig(
        mode=mode,
        n_candidates=n_candidates,
        max_atoms=max_atoms,
        diffusion=DiffusionConfig(n_timesteps=n_timesteps),
    )

    with st.spinner(f"Generating {n_candidates} candidates ({n_timesteps} diffusion steps each)..."):
        sampler = GuidedDiffusionSampler(diffusion_model, scoring_stack, selectivity_profiler)
        candidates = sampler.generate(
            pocket_tensors, gen_cfg, target_name=pdb_id, anti_target_pockets=anti_target_pockets
        )

    mols = [c.mol_or_none for c in candidates]
    st.session_state.mols = mols
    df = profile_candidates(mols)
    df["docking_like_score"] = [c.score for c in candidates]
    st.session_state.candidates_df = df

    st.session_state.selectivity_profiles = None
    if selectivity_profiler is not None and anti_target_pockets:
        pockets_by_name = {pdb_id: pocket_tensors, **anti_target_pockets}
        profiles = {}
        for i, c in enumerate(candidates):
            physchem = torch.zeros(c.coords.shape[0], pocket_tensors["physchem"].shape[-1])
            atom_type_idx = torch.tensor(
                [ATOM_VOCAB.index(e) if e in ATOM_VOCAB else ATOM_VOCAB.index("OTHER") for e in c.elements]
            )
            profiles[f"candidate_{i}"] = selectivity_profiler.profile(
                pdb_id, list(anti_target_pockets.keys()), pockets_by_name, atom_type_idx, physchem, c.coords
            )
        st.session_state.selectivity_profiles = profiles

if st.session_state.candidates_df is not None:
    df = st.session_state.candidates_df
    mols = st.session_state.mols
    valid_mols = [m for m in mols if m is not None]

    st.divider()
    st.subheader("Candidates")

    summary = summarize(df)
    with st.container(horizontal=True):
        st.metric("Valid candidates", f"{summary.get('n_valid', 0)}/{summary.get('n_candidates', 0)}", border=True)
        st.metric("Mean QED", f"{summary.get('mean_qed', 0):.2f}" if summary.get("mean_qed") is not None else "n/a", border=True)
        st.metric("Mean SAscore", f"{summary.get('mean_sa_score', 0):.2f}" if summary.get("mean_sa_score") is not None else "n/a", border=True)
        st.metric("Lipinski pass rate", f"{summary.get('lipinski_pass_rate', 0):.0%}" if summary.get("lipinski_pass_rate") is not None else "n/a", border=True)

    with st.container(border=True):
        st.markdown("**Ranked candidates**")
        display_cols = [c for c in ["candidate_id", "smiles", "docking_like_score", "qed", "sa_score", "logp", "lipinski_pass"] if c in df.columns]
        st.dataframe(df[display_cols].sort_values("docking_like_score", ascending=False), hide_index=True, width="stretch")

    if valid_mols:
        with st.container(border=True):
            st.markdown("**Diversity**")
            stats = scaffold_diversity(valid_mols)
            with st.container(horizontal=True):
                st.metric("Unique scaffolds", stats["n_unique_scaffolds"], border=True)
                st.metric("Scaffold diversity ratio", f"{stats['scaffold_diversity_ratio']:.2f}", border=True)
                st.metric("Internal diversity", f"{internal_diversity(valid_mols):.2f}", border=True)
                st.metric("Uniqueness", f"{uniqueness(valid_mols):.2f}", border=True)

    if st.session_state.get("selectivity_profiles"):
        profiles = st.session_state.selectivity_profiles
        with st.container(border=True):
            st.markdown("**Selectivity report**")
            st.plotly_chart(affinity_heatmap(profiles), width="stretch")

            anti_names = list(next(iter(profiles.values())).offtarget_delta_g.keys())
            score_matrix = np.array(
                [[p.target_delta_g] + [p.offtarget_delta_g[n] for n in anti_names] for p in profiles.values()]
            )
            pareto_idx = pareto_front_indices(score_matrix).tolist()
            scatter_data = {
                "primary_target_dg": score_matrix[:, 0],
                "worst_case_offtarget_dg": score_matrix[:, 1:].min(axis=1) if score_matrix.shape[1] > 1 else score_matrix[:, 0],
            }
            st.plotly_chart(pareto_front_scatter(scatter_data, pareto_idx), width="stretch")
            st.caption(f"{len(pareto_idx)} of {len(profiles)} candidates are Pareto-optimal (primary target vs. worst-case anti-target).")

            st.dataframe(offtarget_risk_table(profiles), hide_index=True, width="stretch")

    if st.button("Generate downloadable HTML report"):
        report_path = generate_report(
            mols,
            pdb_id=pdb_id,
            target_name=pdb_id,
            selectivity_profiles=st.session_state.get("selectivity_profiles"),
            output_path=Path("data_cache") / "report.html",
        )
        st.download_button(
            "Download report.html",
            data=report_path.read_bytes(),
            file_name="pharmaforge_report.html",
            mime="text/html",
        )
elif not run_clicked:
    st.info("Enter a PDB ID in the sidebar and click **Run generation** to get started.")
