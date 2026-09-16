"""Assembles the outputs of every analysis submodule into one self-contained,
publication-ready HTML report (downloadable straight from the dashboard).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem

from pharmaforge_core.analysis.diversity import internal_diversity, novelty_against_reference, scaffold_diversity
from pharmaforge_core.analysis.physchem import profile_candidates, summarize
from pharmaforge_core.analysis.selectivity_report import affinity_heatmap, offtarget_risk_table
from pharmaforge_core.models.selectivity import SelectivityProfile

_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>PharmaForge Candidate Report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, sans-serif; margin: 2rem auto; max-width: 960px; color: #222; }}
h1, h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
th, td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: left; }}
th {{ background: #f5f5f5; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.5rem; margin: 1rem 0; }}
.metric-card {{ background: #fafafa; border: 1px solid #eee; border-radius: 8px; padding: 0.75rem; }}
.metric-card .value {{ font-size: 1.4rem; font-weight: 600; }}
.metric-card .label {{ font-size: 0.75rem; color: #666; }}
</style></head><body>
<h1>PharmaForge Candidate Report</h1>
<p><em>Generated {timestamp} - pocket: {pdb_id}, target: {target_name}</em></p>

<h2>Summary</h2>
<div class="metric-grid">{summary_cards}</div>

<h2>Physicochemical &amp; ADMET Profile</h2>
{physchem_table}

<h2>Diversity &amp; Novelty</h2>
<div class="metric-grid">{diversity_cards}</div>
<p>Top Bemis-Murcko scaffolds: {top_scaffolds}</p>

<h2>Selectivity Report</h2>
{affinity_heatmap_div}
{offtarget_table}

<p style="color:#888; font-size:0.75rem; margin-top:2rem;">
PharmaForge v0.1 - generation, scoring and selectivity numbers in this report come from the
project's own diffusion generator, learned scoring stack and selectivity profiler. See
docs/whitepaper.md for training status and validation caveats before using these numbers
outside of exploratory research.
</p>
</body></html>
"""


def _metric_card(label: str, value) -> str:
    if isinstance(value, float):
        value = f"{value:.3f}"
    return f'<div class="metric-card"><div class="value">{value}</div><div class="label">{label}</div></div>'


def generate_report(
    mols: list[Chem.Mol],
    pdb_id: str,
    target_name: str,
    selectivity_profiles: dict[str, SelectivityProfile] | None = None,
    reference_smiles: list[str] | None = None,
    output_path: str | Path = "report.html",
) -> Path:
    physchem_df = profile_candidates(mols)
    physchem_summary = summarize(physchem_df)
    scaffold_stats = scaffold_diversity(mols)
    diversity_score = internal_diversity(mols)
    novelty = novelty_against_reference(mols, reference_smiles or [])

    summary_cards = "".join(
        _metric_card(k, v)
        for k, v in {
            "candidates": physchem_summary.get("n_candidates", 0),
            "valid": physchem_summary.get("n_valid", 0),
            "mean QED": physchem_summary.get("mean_qed"),
            "mean SAscore": physchem_summary.get("mean_sa_score"),
            "Lipinski pass rate": physchem_summary.get("lipinski_pass_rate"),
        }.items()
        if v is not None
    )
    diversity_cards = "".join(
        _metric_card(k, v)
        for k, v in {
            "unique scaffolds": scaffold_stats["n_unique_scaffolds"],
            "scaffold diversity ratio": scaffold_stats["scaffold_diversity_ratio"],
            "internal diversity": diversity_score,
            "novelty rate": novelty.get("novelty_rate"),
        }.items()
        if v is not None
    )

    affinity_div = ""
    offtarget_html = ""
    if selectivity_profiles:
        fig = affinity_heatmap(selectivity_profiles)
        affinity_div = fig.to_html(full_html=False, include_plotlyjs="cdn")
        offtarget_df = offtarget_risk_table(selectivity_profiles)
        offtarget_html = offtarget_df.to_html(index=False)
    else:
        offtarget_html = "<p><em>No anti-targets configured for this run.</em></p>"

    html = _TEMPLATE.format(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        pdb_id=pdb_id,
        target_name=target_name,
        summary_cards=summary_cards or "<p>No valid candidates.</p>",
        physchem_table=physchem_df.to_html(index=False) if not physchem_df.empty else "<p>No candidates.</p>",
        diversity_cards=diversity_cards,
        top_scaffolds=", ".join(f"{s} ({c})" for s, c in scaffold_stats["top_scaffolds"][:5]) or "n/a",
        affinity_heatmap_div=affinity_div,
        offtarget_table=offtarget_html,
    )

    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
    return output_path
