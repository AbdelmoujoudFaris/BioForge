# PharmaForge

**A structure-aware, 3D-native de novo drug design framework.** Generates
molecules as 3D atomic coordinates directly inside a protein binding pocket
using an E(3)-equivariant diffusion model, scores them with a differentiable
neural surrogate (no external docking calls in the loop), and optimizes for
**selectivity** against an arbitrary panel of anti-targets as a first-class,
generation-time objective.

> **Status: research-stage framework, not yet trained on real data.** The
> full architecture, data pipeline, training scripts, analysis suite, and
> dashboard are implemented and unit-tested (35 tests passing). No
> checkpoint trained on PDBbind/CrossDocked2020 ships with this repository -
> that requires a GPU and a multi-GB dataset download this project wasn't
> built with access to. **Read [`docs/whitepaper.md`](docs/whitepaper.md)
> before citing any generated molecule or benchmark number** - it states
> plainly what has and hasn't been empirically validated.

## Why not another AlphaDrug-style pipeline?

| | AlphaDrug | PharmaForge |
|---|---|---|
| Generation | SMILES-token autoregressive (Lmser Transformer) | 3D E(3)-equivariant diffusion, atoms placed directly in the pocket |
| Protein conditioning | Sequence only | Full 3D pocket graph (coordinates + physicochemical features) |
| Scoring | External SMINA docking call per MCTS rollout | Differentiable neural surrogate, milliseconds/call, gradient w.r.t. coordinates |
| Search | Vanilla discrete MCTS | Gradient-guided diffusion sampling + differentiable critic-scored tree search |
| Objective | Single target affinity | **Multi-objective selectivity** against an arbitrary anti-target panel, with a reported selectivity index per molecule |

See [`docs/architecture.md`](docs/architecture.md) for the full module map and data-flow diagram.

## Install

```bash
git clone <repo-url> && cd PharmaForge
pip install -e .
```

Optional extras: `pip install -e ".[dev]"` (tests/lint), `".[retrosynthesis]"`
(AiZynthFinder), `".[visualization]"` (NGLview/MDAnalysis for notebooks).
`fpocket` (blind pocket detection) is a separate system binary - see
`docs/architecture.md`; without it, pocket resolution falls back to a
geometric-center heuristic.

## Quickstart

```bash
streamlit run pharmaforge/ui/dashboard.py
```

![PharmaForge dashboard demo](assets/pharmaforge_dashboard_demo.gif)

*Demo run against an untrained checkpoint (the dashboard's own banner says
so) - it demonstrates the pipeline (pocket resolution → generation → ranked
candidates → diversity metrics), not optimized molecules. See the caveat
above.*

Type a PDB ID (e.g. `6LU7`), optionally a reference ligand residue name and
a list of anti-target PDB IDs for selectivity-aware design, click **Run
generation**, and browse ranked candidates, diversity metrics, the
selectivity heatmap/Pareto front, and a downloadable HTML report. Without a
checkpoint directory containing real trained weights, the dashboard runs the
exact same code against randomly-initialized weights in a clearly labeled
demo mode - useful for verifying the pipeline, not for generating real
candidates.

Programmatic use:

```python
from pharmaforge.generation.pipeline import PharmaForgePipeline
from pharmaforge.models.diffusion import LigandDiffusionModel
from pharmaforge.models.scoring import ScoringStack
from pharmaforge.config import DiffusionConfig, GenerationConfig, ScoringConfig

pipeline = PharmaForgePipeline(
    diffusion_model=LigandDiffusionModel(DiffusionConfig()),
    scoring_stack=ScoringStack(ScoringConfig()),
)
candidates = pipeline.run(
    pdb_id="6LU7",
    cfg=GenerationConfig(mode="de_novo", n_candidates=20),
    anti_target_pdb_ids={"hERG": "5VA1"},  # selectivity-aware generation
)
```

## Training

```bash
python -m pharmaforge.training.train_diffusion --data-root <CrossDocked2020 path>
python -m pharmaforge.training.train_scoring --data-root <PDBbind path> --affinity-index INDEX_general_PL_data.2020
python -m pharmaforge.training.train_actor_critic --scoring-checkpoint checkpoints/scoring/scoring.pt
```

Every script also supports `--synthetic` to smoke-test the training loop
without downloading a real dataset - what CI runs.

## Benchmarking

```bash
python -m pharmaforge.benchmarks.run_benchmark --pdb-ids 6LU7 --checkpoint-dir checkpoints
```

Prints PharmaForge's own metrics side by side with a literature reference
table for Pocket2Mol/TargetDiff/DiffSBDD (see `benchmarks/baselines.py` for
sourcing caveats - verify before citing). Without `--checkpoint-dir`
pointing at real trained weights, the script says so loudly rather than
producing a misleading comparison.

## Docker

```bash
docker compose up --build
```

Serves the dashboard at `http://localhost:8501`, with `data_cache/` and
`checkpoints/` mounted as volumes.

## Tests

```bash
pytest tests/ -q
```

## Repository layout

```
pharmaforge/
├── config.py        data/           models/          generation/
├── scoring/         analysis/       training/         benchmarks/
└── ui/
tests/               notebooks/      docs/             scripts/
```

See [`docs/architecture.md`](docs/architecture.md) for what each module does
and why.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) - module map, data-flow diagram, design rationale.
- [`docs/whitepaper.md`](docs/whitepaper.md) - full technical report: architecture, the Selectivity Profiler, training protocol, and honest benchmark status.
- [`notebooks/`](notebooks) - quickstart, pocket-conditioned generation, selectivity profiler, and benchmark-analysis tutorials.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) - dev setup and contribution guidelines.

## License

MIT - see [`LICENSE`](LICENSE).
