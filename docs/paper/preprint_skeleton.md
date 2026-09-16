# BioForge: An Open-Source, Modular Platform for AI-Native Drug Repurposing and De Novo Design with Causal Mechanistic Interpretability

*arXiv preprint skeleton - fill in bracketed sections, add real figures/results before submission.*

## Abstract

[150-250 words. State the gap (single-model SBDD pipelines report a docking
score with no systems-level explanation of *why* a candidate should work),
BioForge's architectural answer (modular microservices + multi-objective
generation ensemble + a causal mechanistic interpretability layer over real
PPI/pathway data), and what's empirically validated vs. architecturally
demonstrated in this release - do not claim benchmark numbers that
`benchmarks/` hasn't actually produced.]

## 1. Introduction

- The drug discovery bottleneck this targets: [target-to-hypothesis latency /
  lack of mechanistic explainability in generative SBDD / whatever the
  actual motivating gap is for your deployment].
- Prior work: SMILES-autoregressive SBDD (AlphaDrug-style), 3D diffusion SBDD
  (DiffSBDD, Pocket2Mol, TargetDiff), docking-in-the-loop search.
- Contribution list:
  1. A modular, independently-deployable microservices architecture for the
     full target-to-report pipeline (Section 3).
  2. A genuine multi-objective (Pareto) ranking over an extensible generator
     ensemble, not a single scalar docking score (Section 3.2).
  3. **Causal mechanistic interpretability**: perturbation-graph construction
     over real PPI data, network-propagated pathway analysis, and a
     confidence-scored, literature-anchored causal chain per candidate
     (Section 4) - to our knowledge not offered by [cite prior SBDD
     platforms you're differentiating against].
  4. A fully reproducible, open-source reference implementation with
     documented honesty boundaries between real and stub components
     (Section 6, `docs/whitepaper.md`).

## 2. Related Work

[DiffSBDD, DiffDock/DiffDock-L, Pocket2Mol, TargetDiff, AlphaDrug; network-
propagation methods for perturbation biology (NetICS, RWR-based
repurposing); systems pharmacology / polypharmacology platforms.]

## 3. System Architecture

### 3.1 Structure Intelligence
[Pocket ensemble: classical grid scan x geometric-DL re-ranking. Cite
LIGSITE (Hendlich et al., 1997) for the classical method's lineage.]

### 3.2 Generative Chemistry Ensemble
[E(3)-equivariant diffusion architecture (cite the EGNN backbone lineage:
Satorras et al., 2021); multi-objective ranking methodology - non-dominated
sorting over affinity/selectivity/SA/ADMET.]

### 3.3 Physics-Based Refinement
[Differentiable scoring stack architecture; MC-dropout uncertainty (Gal &
Ghahramani, 2016); the elastic-network OpenMM equilibration and its
relationship to a full GAFF2/OpenFF treatment.]

## 4. Causal Mechanistic Interpretability & Systems Pharmacology

[This is the section to expand most for a real submission. Cover:
perturbation-graph construction methodology over STRING/OmniPath; the
network-propagation formulation (personalized PageRank / RWR) and its
relationship to - and explicit gap from - a rigorous causal (do-calculus)
estimate; differential pathway analysis against KEGG; how literature support
counts are computed and their limitations (a count, not a relevance-weighted
evidence score); the confidence-scoring formulation for the emitted causal
chain, and a worked example.]

## 5. Evaluation

[**Do not fill this in with fabricated numbers.** Run `benchmarks/` for real
against DUD-E/LIT-PCBA/your custom holdout set, report exactly what
came out, and state plainly whether the checkpoint used was trained or the
randomly-initialized demo weights described in `docs/whitepaper.md`. If no
trained checkpoint exists yet, say that explicitly rather than reporting
benchmark numbers from an untrained model as if they were meaningful.]

## 6. Limitations and Honesty Boundaries

[Summarize `docs/whitepaper.md`'s table: what's real vs. a documented
extension point, stage by stage. A reviewer will find the actual repository
regardless - state this up front rather than have it discovered.]

## 7. Reproducibility Statement

[Provenance tracking (MLflow/local JSONL), versioned engine
(`engines/pharmaforge_core`), the test suite and its coverage, and exact
commands to reproduce any reported number.]

## References

[BibTeX/citations for: EGNN/E(3)-equivariant GNNs, diffusion models for
molecule generation, DiffSBDD/Pocket2Mol/TargetDiff/DiffDock, LIGSITE,
network propagation in perturbation biology, STRING/OmniPath/KEGG/PubMed as
data sources, NSGA-III/MOEA-D multi-objective optimization, Monte Carlo
dropout uncertainty.]
