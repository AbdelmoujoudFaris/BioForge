"""Reference numbers for published structure-based drug design baselines.

IMPORTANT HONESTY NOTE: the figures below are commonly-cited approximate
values from each paper's own CrossDocked2020 test-set results table,
reconstructed from memory for convenience during development. They are
**not** re-derived by running those models in this repository (none of
AlphaDrug/Pocket2Mol/DiffSBDD/TargetDiff are vendored here), and small
digits may be misremembered. Before using this table in any report,
publication, or decision, verify every cell against the cited paper's own
table. Treat this module as a scaffold for a real comparison, not as a
citable result in itself.

  * AlphaDrug   - Qian et al., "AlphaDrug: Protein Target Specific De Novo
                  Molecular Generation", PNAS Nexus 2022 (fragment-based
                  SMILES generation + MCTS + SMINA docking).
  * Pocket2Mol  - Peng et al., "Pocket2Mol: Efficient Molecular Sampling
                  Based on 3D Protein Pockets", ICML 2022.
  * DiffSBDD    - Schneuing et al., "Structure-based Drug Design with
                  Equivariant Diffusion Models", 2022/2023 (arXiv, later
                  Nature Computational Science 2024).
  * TargetDiff  - Guan et al., "3D Equivariant Diffusion for Target-Aware
                  Molecule Generation and Affinity Prediction", ICLR 2023.

All four report on the CrossDocked2020 test set (100 pockets, ~100
molecules/pocket), with Vina Score/Min/Dock in kcal/mol (lower = better),
QED and SA in [0, 1] (higher = better for both, using the 1-normalized SA
convention rather than the raw 1-10 SAscore), and Diversity as mean internal
Tanimoto distance.
"""
from __future__ import annotations

REFERENCE_METRICS = {
    # method: {metric: approximate_value_from_paper}
    "AlphaDrug": {
        "vina_score": None,  # AlphaDrug reports SMINA docking score on its own pocket set (CrossDocked2020-derived, not the exact same split as the others) - not directly comparable without re-docking on a shared split.
        "qed": None,
        "sa": None,
        "diversity": None,
        "source": "Qian et al., PNAS Nexus 2022 - metrics reported on a different benchmark split; re-run on shared CrossDocked2020 split before comparing directly.",
    },
    "Pocket2Mol": {
        "vina_score": -5.14,
        "vina_min": -6.42,
        "vina_dock": -7.15,
        "qed": 0.56,
        "sa": 0.74,
        "diversity": 0.69,
        "source": "Peng et al., ICML 2022, Table 1 (approximate, verify against source)",
    },
    "TargetDiff": {
        "vina_score": -5.47,
        "vina_min": -6.64,
        "vina_dock": -7.80,
        "qed": 0.48,
        "sa": 0.58,
        "diversity": 0.72,
        "source": "Guan et al., ICLR 2023, Table 1 (approximate, verify against source)",
    },
    "DiffSBDD": {
        "vina_score": None,
        "vina_min": None,
        "vina_dock": -6.95,
        "qed": 0.47,
        "sa": 0.60,
        "diversity": 0.73,
        "source": "Schneuing et al., 2023, self-reported metrics (approximate, verify against source)",
    },
    "Test set (reference ligands)": {
        "vina_score": -6.36,
        "vina_min": -6.71,
        "vina_dock": -7.45,
        "qed": 0.48,
        "sa": 0.73,
        "diversity": None,
        "source": "CrossDocked2020 co-crystallized ligands, as reported in the TargetDiff/Pocket2Mol comparison tables",
    },
}


def reference_table_markdown() -> str:
    methods = list(REFERENCE_METRICS.keys())
    metrics = ["vina_score", "vina_min", "vina_dock", "qed", "sa", "diversity"]
    header = "| method | " + " | ".join(metrics) + " |"
    sep = "|---" * (len(metrics) + 1) + "|"
    rows = [header, sep]
    for method in methods:
        values = REFERENCE_METRICS[method]
        row = "| " + method + " | " + " | ".join(
            (f"{values[m]:.2f}" if values.get(m) is not None else "n/a") for m in metrics
        ) + " |"
        rows.append(row)
    return "\n".join(rows)
