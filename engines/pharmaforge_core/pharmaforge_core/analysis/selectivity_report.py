"""Selectivity reporting: affinity heatmap across a target panel, per-molecule
selectivity radar charts, and an off-target risk table. Built on Plotly so
the same figure objects drop straight into the Streamlit dashboard
(`ui/dashboard.py`) or export to static HTML/PNG for the whitepaper.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pharmaforge_core.models.selectivity import SelectivityProfile


def affinity_heatmap(profiles: dict[str, SelectivityProfile], molecule_labels: list[str] | None = None) -> go.Figure:
    """profiles: {molecule_id: SelectivityProfile}. Rows = molecules, columns
    = [primary target, *anti-targets], cells = predicted ΔG (kcal/mol).
    """
    mol_ids = list(profiles.keys())
    labels = molecule_labels or mol_ids
    anti_target_names = list(next(iter(profiles.values())).offtarget_delta_g.keys()) if profiles else []
    columns = ["primary_target"] + anti_target_names

    data = []
    for mol_id in mol_ids:
        p = profiles[mol_id]
        data.append([p.target_delta_g] + [p.offtarget_delta_g[n] for n in anti_target_names])

    df = pd.DataFrame(data, index=labels, columns=columns)
    fig = px.imshow(
        df,
        color_continuous_scale="RdBu",
        aspect="auto",
        labels=dict(color="predicted ΔG (kcal/mol)"),
        title="Predicted binding affinity across target panel",
    )
    return fig


def selectivity_radar(profile: SelectivityProfile, molecule_label: str = "candidate") -> go.Figure:
    categories = ["primary_target"] + list(profile.offtarget_delta_g.keys())
    # Plot selectivity index (log scale) so "very selective" reads as a big
    # spike on the primary target axis relative to every anti-target.
    values = [1.0] + [profile.selectivity_index.get(name, 0.0) for name in categories[1:]]
    log_values = [max(v, 1e-6) for v in values]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(r=log_values, theta=categories, fill="toself", name=molecule_label)
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(type="log", visible=True)),
        title=f"Selectivity profile - {molecule_label}",
        showlegend=True,
    )
    return fig


def offtarget_risk_table(profiles: dict[str, SelectivityProfile], si_floor: float = 10.0) -> pd.DataFrame:
    rows = []
    for mol_id, profile in profiles.items():
        for anti_target, si in profile.selectivity_index.items():
            rows.append(
                {
                    "molecule": mol_id,
                    "anti_target": anti_target,
                    "selectivity_index": si,
                    "risk": "high" if si < si_floor else "low",
                }
            )
    return pd.DataFrame(rows).sort_values("selectivity_index")


def pareto_front_scatter(candidate_scores: dict, pareto_indices: list[int]) -> go.Figure:
    """2D projection (primary target vs. worst-case anti-target ΔG) of a
    multi-objective search result, with Pareto-optimal points highlighted -
    what the dashboard's candidate browser uses to let users click through
    the Pareto front interactively.
    """
    df = pd.DataFrame(candidate_scores)
    df["pareto_optimal"] = df.index.isin(pareto_indices)
    fig = px.scatter(
        df,
        x="primary_target_dg",
        y="worst_case_offtarget_dg",
        color="pareto_optimal",
        title="Multi-objective Pareto front: primary target vs. worst-case anti-target",
        labels={"primary_target_dg": "Primary target ΔG (kcal/mol)", "worst_case_offtarget_dg": "Worst anti-target ΔG (kcal/mol)"},
    )
    return fig
