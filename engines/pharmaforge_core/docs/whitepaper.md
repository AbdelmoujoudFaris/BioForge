# PharmaForge: A Structure-Aware De Novo Drug Design Framework

*Technical report - v0.1*

## Abstract

PharmaForge is an open-source framework for pocket-conditioned de novo
molecule generation built around three ideas: (1) generate ligands directly
as 3D atomic coordinates inside a binding pocket using an E(3)-equivariant
diffusion model, rather than as 1D SMILES tokens; (2) replace external
docking calls in the search loop with a differentiable neural surrogate that
predicts binding affinity, pose validity, and interaction fingerprints in a
single millisecond-scale forward pass; and (3) make multi-objective
selectivity - maximizing affinity for a primary target while minimizing it
for a panel of anti-targets - a first-class part of generation, not a
post-hoc filter. This report describes the architecture, the training
protocol needed to reach a production-quality checkpoint, and - critically -
what has and has not been empirically validated in this repository as of
this writing.

**Read this before citing any number from this framework**: the codebase
implements the full pipeline described in the project's design brief and is
covered by unit tests (`tests/`, 35 passing) that verify each component's
correctness in isolation (E(3) equivariance, gradient flow through the
scoring stack, selectivity-index arithmetic, the multi-objective search, the
analysis suite). It has **not** been trained end-to-end on PDBbind or
CrossDocked2020, because doing so requires a multi-GB dataset download and
GPU-days of compute that this development environment does not have. Every
number in the "Benchmark status" section below is labeled accordingly.

## 1. Architecture

### 1.1 Pocket-conditioned 3D diffusion generator

`models/diffusion.py` implements a standard DDPM (Ho et al., 2020) noise
schedule (`GaussianDiffusionSchedule`, cosine or linear) over ligand atom
coordinates, with atom-type logits regressed jointly by the same denoising
network (`LigandDenoiser`). The denoiser is `models/egnn.py`'s
`PocketConditionedEGNN`: an E(3)-equivariant graph neural network (Satorras
et al., 2021 EGNN update rule) operating on the combined pocket+ligand
graph, where pocket atom coordinates are held fixed (rigid receptor) and
only ligand atom coordinates are updated at each message-passing layer.
Center-of-mass-free noise (translation-invariant) is used during training
and sampling, following Hoogeboom et al.'s E(3) equivariant diffusion for
molecules.

Verified: forward/backward passes run correctly; rotating+translating the
input pocket produces an identically transformed ligand output
(`tests/test_egnn.py`); `training_loss` decreases parameters under one SGD
step (`tests/test_diffusion.py`). **Not verified**: sample quality after a
real training run - an untrained model produces chemically implausible
output (confirmed interactively: an untrained checkpoint generates
low-diversity, low-Lipinski-pass-rate molecules, exactly as expected of
random weights; see the dashboard's own runtime warning when no checkpoint
is found).

### 1.2 Differentiable scoring stack

`models/scoring.py`'s `ScoringStack` shares the same EGNN backbone and adds
three heads: binding affinity (a ΔG-like scalar), a clash/pose-validity
logit, and a per-pocket-atom × per-ligand-atom interaction-fingerprint
tensor (6 channels: H-bond donor/acceptor, hydrophobic, π-stacking, halogen
bond, salt bridge). `differentiable_score` combines the first two into a
single scalar usable as (a) classifier-style guidance inside the diffusion
sampler's reverse process, and (b) the critic value in the actor-critic
search - replacing AlphaDrug's per-rollout SMINA docking call, which is
orders of magnitude slower and non-differentiable.

Verified: gradients flow correctly from the scalar score back to ligand
coordinates (`tests/test_scoring.py`); a full guided-sampling loop
(`GuidedDiffusionSampler`) runs end-to-end on both synthetic pockets and a
real PDB structure (6LU7, SARS-CoV-2 main protease), producing valid RDKit
molecules from raw diffusion output. **Not verified**: correlation between
predicted and experimental ΔG - this requires training on labeled
PDBbind/CrossDocked2020 affinity data (`training/train_scoring.py` implements
this training loop, including a PDBbind `INDEX_general_PL_data` parser, but
has only been smoke-tested against synthetic random targets in this repo).

### 1.3 Graph-transformer actor-critic and differentiable tree search

`models/actor_critic.py`'s `GraphTransformerActor` proposes atom-addition
actions (new atom type, attachment point, bond type) conditioned on the
current partial molecule embedded jointly with the pocket. Rather than
vanilla MCTS with random rollouts, `DifferentiableTreeSearch` expands a beam
of candidate partial molecules at each depth and ranks them with a single
batched forward pass through the (differentiable) critic - no rollout
simulation, no docking call. `training/train_actor_critic.py` trains the
actor via REINFORCE against a frozen, pretrained scoring-stack critic, using
a running-mean baseline for variance reduction.

Verified: action sampling, advantage computation, and beam search all run
correctly (`tests/test_actor_critic.py`); a real bug in bond-type sampling
(it was sampling one bond type per existing atom instead of one for the
chosen attachment point) was caught by the REINFORCE training smoke test
and fixed during development. **Not verified**: policy quality after
extended training - the training script currently draws pockets from
`SyntheticPocketLigandDataset` by default; production training needs a real
pocket dataset wired into the `--n-pockets` pool.

### 1.4 Selectivity Profiler - the signature feature

Described in full in `docs/architecture.md`. The key design decision: rather
than bolting selectivity on as a post-generation filter (generate N
molecules, throw away the ones that also bind off-targets), the anti-target
penalty is injected as a guidance gradient *during* diffusion sampling
(`SelectivityProfiler.constrained_guidance_score`), so the reverse diffusion
process is steered away from off-target-favorable geometries at every
denoising step. A separate, population-level Pareto analysis
(`pareto_front_indices`, non-dominated sorting; `run_multi_objective_search`,
a full NSGA-III/MOEA-D formulation via `pymoo`) lets users browse the
resulting trade-off surface after generation.

Verified end-to-end through the dashboard: generating 8 candidates for
6LU7 with 1CRN configured as an anti-target correctly produced a selectivity
index per candidate, an affinity heatmap, and a Pareto-front scatter (7/8
candidates were Pareto-optimal in that specific untrained-weights run - the
number itself is not meaningful, but the full plumbing - profiler → ensemble
→ heatmap → Pareto sort → dashboard render → HTML report - is confirmed
working).

## 2. Training protocol for a production checkpoint

1. Download PDBbind (refined + general sets) and/or CrossDocked2020 and
   extract to the `<complex_id>/<complex_id>_{protein,ligand}.{pdb,sdf}`
   layout `data/datasets.py` expects.
2. `python -m pharmaforge.training.train_diffusion --data-root <path> --dataset crossdocked2020 --epochs 100`
3. `python -m pharmaforge.training.train_scoring --data-root <path> --affinity-index INDEX_general_PL_data.2020`
4. `python -m pharmaforge.training.train_actor_critic --scoring-checkpoint checkpoints/scoring/scoring.pt`
5. Re-run `python -m pharmaforge.benchmarks.run_benchmark --checkpoint-dir checkpoints` and compare against
   `benchmarks/baselines.py`'s reference table.

Expect step 2 alone to need a multi-day run on a single modern GPU for a
CrossDocked2020-scale dataset (100k+ complexes), based on published training
times for comparably-sized equivariant diffusion SBDD models (TargetDiff,
DiffSBDD).

## 3. Benchmark status

`benchmarks/baselines.py` contains commonly-cited (reconstructed from memory,
explicitly flagged as "verify before citing") reference numbers for
Pocket2Mol, TargetDiff, and DiffSBDD on CrossDocked2020, and marks AlphaDrug
as not directly comparable without re-docking on a shared split.
`benchmarks/run_benchmark.py` runs PharmaForge's own generation + analysis
pipeline over a set of pockets and reports the same metric family (a learned
score proxy in place of Vina score/min/dock - the two are not the same
units and should not be conflated - plus QED, SA, diversity, validity).

**As of this report, that script has only been run against randomly
initialized weights** (no GPU training run was performed), and it says so
explicitly in its own output (`source: this run (UNTRAINED WEIGHTS - not
comparable)`). We are not claiming PharmaForge outperforms AlphaDrug,
Pocket2Mol, DiffSBDD, or TargetDiff on any metric. The architecture and
harness are complete and tested; the empirical comparison requires the
training run described in Section 2, which is future work for whoever runs
this on real GPU infrastructure.

## 4. What is genuinely novel here vs. AlphaDrug

- 3D-native, pocket-graph-conditioned generation instead of sequence-conditioned SMILES generation.
- A learned, differentiable, millisecond-scale scoring surrogate instead of an external per-rollout docking call.
- Gradient-guided diffusion sampling and a differentiable-critic tree search instead of discrete MCTS with rollout-based value estimation.
- A first-class, generation-time multi-objective Selectivity Profiler with a reported per-molecule selectivity index against an arbitrary anti-target panel - no equivalent exists in AlphaDrug.

## 5. Performance notes

Guided-diffusion sampling currently runs roughly 1-1.5 seconds per denoising
step on CPU for a ~230-atom pocket+ligand graph at demo scale (hidden_dim=32,
3 EGNN layers) - noticeably slower than the raw model size would suggest.
Two un-optimized costs dominate: (1) every guided step re-runs *two* full
EGNN forward passes (the denoiser, then a second pass inside the scoring
stack's guidance call) plus a backward pass through the second, and (2) the
pocket-pocket portion of the radius graph is identical at every timestep
(only ligand coordinates move) but is currently recomputed from scratch each
step rather than cached. Candidates are also generated one at a time rather
than batched. None of this affects correctness (verified in `tests/`), but a
GPU deployment should batch candidates together and cache the static
pocket-pocket subgraph before relying on wall-clock throughput numbers.

## 6. Honest limitations

- No pretrained checkpoints ship with this repository (see Section 2 for the training recipe).
- `RuleBasedADMET` is a descriptor heuristic, not a trained ADMET model (`scoring/admet.py` documents `LearnedADMET` as the extension point).
- `RetrosynthesisScorer` falls back to the RDKit SAscore heuristic unless a real AiZynthFinder config/model is supplied.
- Benchmark comparisons against published baselines are not yet empirically re-derived in this repository.
