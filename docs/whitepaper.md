# BioForge Whitepaper: What's Real, What's a Stub, and Why

This document exists because the platform spec BioForge implements names a
long list of specific tools (Boltz-1, DiffSBDD, GNINA, FEP+, DoWhy,
Equiformer v2, ...), and it would be easy - and dishonest - to build a demo
that name-drops all of them without actually wiring any in. This is the
single place that states, stage by stage, exactly what runs for real versus
what is a documented, unimplemented extension point. Cross-reference it
against `src/bioforge/core/*` docstrings, which repeat the same caveats
inline.

## 1. Intake & Structure Intelligence (`core/structure_intel.py`)

| Piece | Status |
|---|---|
| PDB ID intake | **Real.** RCSB download via the vendored `pharmaforge_core.data.pdb_download`. |
| AlphaFold DB intake | **Real.** Live `alphafold.ebi.ac.uk` API call, including mean pLDDT parsed from the downloaded structure. |
| UniProt ID intake | **Real.** Live UniProt REST sequence fetch + AlphaFold DB structure lookup. |
| Raw sequence intake (de novo structure prediction) | **Not implemented.** `StructurePredictorAdapter`/`Boltz1Adapter`/`Chai1Adapter`/`AlphaFold3APIAdapter` raise `NotImplementedError` - none have GPU weights or a paid API key wired in. |
| Classical pocket detection | **Real**, two paths: the real `fpocket` binary if installed on `PATH` (via the vendored engine), else a bounded LIGSITE-style grid scan implemented in this repository (`ligsite_grid_pockets`) - a real, if coarse and radius-bounded, protein-solvent-protein-event detector, not a call to a real geometric-DL tool like PocketMiner/FTMap ML. |
| Geometric-deep-learning pocket scoring | **Real architecture, untrained weights.** `PocketDruggabilityGNN` is a genuine small `torch_geometric` message-passing network run on each candidate pocket's real atom graph, with fixed-seed, randomly-initialized weights - it demonstrates the ensemble mechanics, not a trained druggability predictor. |

## 2. Generative Chemistry Engine (`core/generation_ensemble.py`)

| Piece | Status |
|---|---|
| E(3)-equivariant pocket-conditioned diffusion | **Real, untrained.** The vendored, tested (`engines/pharmaforge_core/tests/test_egnn.py` verifies equivariance directly) architecture, with the same "no PDBbind/CrossDocked2020 checkpoint ships here" caveat as upstream. |
| DiffSBDD / DiffDock-L / fine-tuned latent diffusion | **Not implemented.** Documented adapter classes raising `NotImplementedError`; the ensemble skips them rather than failing the request. |
| Multi-objective Bayesian optimization / Pareto ranking | **Real.** Genuine non-dominated sort (via the vendored engine's `pareto_front_indices`, the same routine backing its Selectivity Profiler) over affinity, QED, synthetic accessibility, and an ADMET-desirability proxy - real for whatever candidates the enabled generators actually produced. |
| Selectivity-aware generation (anti-target panel) | **Real.** Reuses the vendored `SelectivityProfiler`/`MultiTargetScoringEnsemble` for constrained-guidance sampling during generation. |

## 3. Physics-Based Refinement (`core/refinement.py`)

| Piece | Status |
|---|---|
| MD equilibration | **Real OpenMM run** (minimization + short Langevin dynamics), but against a generic elastic-network force field built from RDKit-perceived bonds/angles + per-element LJ parameters + Gasteiger charges - **not** a validated GAFF2/OpenFF parameterization (that needs an external AM1-BCC charge toolchain this environment doesn't reliably have; `LearnedForceField` documents the gap). |
| Pose prediction/refinement (GNINA/DiffDock substitute) | **Real, but a substitute, not the named tools.** Gradient ascent on the vendored differentiable `ScoringStack`'s score w.r.t. ligand coordinates - genuinely differentiable local refinement, explicitly *not* a GNINA docking run or a DiffDock inference pass (`GninaPoseAdapter`/`DiffDockPoseAdapter` document those gaps). |
| Uncertainty quantification | **Real Monte Carlo dropout** (Gal & Ghahramani, 2016) - dropout kept active at inference, sampled N times. This is epistemic uncertainty from one checkpoint's dropout masks, not from an ensemble of independently trained models. |
| Metadynamics / FEP+ | **Not implemented.** `FEPAdapter` documents the gap: needs a validated force field and an ns-scale simulation budget no request handler can spend. |

## 4. Database Retrieval & Similarity (`core/retrieval.py`)

| Piece | Status |
|---|---|
| ECFP4 / Tanimoto | **Real** (RDKit Morgan fingerprints, radius 2, 2048 bits). |
| PubChem similarity search | **Real, live, keyless API.** Verified working during development. |
| ChEMBL similarity search | **Real, live API call** with a graceful fallback - the public endpoint was observed returning 5xx independently of this code during development; failures don't crash the request. |
| ZINC search | **Attempted real call, unverified contract.** Falls back to a small, clearly-labeled bundled sample on any failure. See the function's docstring before relying on the live path. |
| DrugBank search | **Documented stub without a license key.** Returns a small, clearly-labeled bundled sample; wire `BIOFORGE_DRUGBANK_API_KEY` and real API access to make it live. |
| "3D equivariant embedding" (Equiformer v2/MACE stand-in) | **Real forward pass, untrained weights.** Uses the vendored E(3)-equivariant EGNN backbone (verified-equivariant architecture) with fixed-seed random weights - not Equiformer v2 or MACE, and not trained. |
| Pharmacophore matching | **Real, simplified.** RDKit chemical-feature perception reduced to a feature-count-vector cosine similarity, not a full 3D pharmacophore alignment engine. |
| MHFP fingerprints / contrastive embedding alignment | **Not implemented.** `MHFPEncoder`/`ContrastiveAligner` document the gaps (a separate package; paired multi-modal training data, respectively). |

## 5. ADMET & Safety (`core/admet_safety.py`)

| Piece | Status |
|---|---|
| Absorption/BBB/CYP/hERG heuristics | **Real**, the vendored `RuleBasedADMET` - descriptor-based (BOILED-Egg-style) heuristics, not a trained model. |
| Off-target panel screening | **Real**, structure-based, using the same differentiable scoring stack against a small, real anti-target pocket panel - a request-time-feasible stand-in for genuinely proteome-wide docking, not proteome-wide itself. |
| DeepTox / DeepConv-DTI / TransformerCPI | **Not implemented.** `DeepToxAdapter`/`DTIAdapter` document the gaps (need Tox21/ToxCast training or a pretrained checkpoint, respectively). |

## 6. Causal Mechanistic Interpretability & Systems Pharmacology (`core/mechanism.py`)

See `docs/architecture.md`'s dedicated section. In short: perturbation-graph
construction (STRING/OmniPath) and downstream-pathway analysis (KEGG) are
real, live network calls; network propagation (personalized PageRank) is a
real, standard systems-biology technique, but an explicitly *weaker* proxy
for a rigorous causal estimate than `DoWhyCausalRefinement` (not
implemented) would provide; literature support counts are real PubMed
E-utilities searches; narrative generation is a real, deterministic offline
template by default, or a real Anthropic API call (true RAG over the
computed chain) when `BIOFORGE_LLM_PROVIDER`/`BIOFORGE_LLM_API_KEY` are set.

## 7. Advanced Analysis & Reporting (`core/reporting.py`)

| Piece | Status |
|---|---|
| Synthetic tractability | **Real.** SAscore always; real AiZynthFinder if installed and configured with a stock/policy file (neither is vendored - multi-GB downloads). |
| Patent landscape | **Real, limited.** A best-effort PubChem patent-annotation cross-reference check, not a full-text claims search. `SureChEMBLAdapter` documents the gap for real SureChEMBL API access. |
| Clinical Translation Score | **Real, transparent composite**, documented as a starting weighted allocation across four evidence classes (`_DEFAULT_WEIGHTS`) - not a validated clinical-success predictor. Recalibrate before using it for anything beyond triage/ranking. |
| Uncertainty flags | **Real**, surfacing the genuine MC-dropout std and any per-field threshold from every other stage. |
| Polypharmacology network | **Real data**, rendered with Plotly/NetworkX in the dashboard rather than Cytoscape.js/D3.js (functionally equivalent for this deployment; swap the rendering layer, not the data, if you specifically need one of those libraries). |

## Reproducibility

Every service call wraps its work in `bioforge.common.provenance.track_run`,
logging params/metrics/artifact-refs to MLflow when reachable, and always to
a local `.provenance/<stage>.jsonl` trail regardless - a pipeline run's
provenance survives even without an MLflow server. The vendored engine
(`engines/pharmaforge_core`) ships its own benchmark harness against
DUD-E/LIT-PCBA baselines (with the same "verify before citing" caveat on its
literature reference numbers); `benchmarks/` extends it with BioForge-level
custom holdouts.
