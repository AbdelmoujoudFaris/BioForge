# BioForge

**The open standard for AI-native drug repurposing and de novo design.**
A modular, containerized microservices platform - not a single notebook, not
a monolith - that takes a protein target from a PDB ID/UniProt accession/raw
sequence all the way through structure resolution, ensemble generative
chemistry, physics-based refinement, multi-database retrieval, ADMET/safety
screening, and out to BioForge's differentiating feature: a **causal
mechanistic interpretability engine** that turns a generated candidate into
an explained, literature-anchored systems-pharmacology hypothesis instead of
a bare docking score.

> **Status: research-stage, and honest about it.** Every stage in this
> repository actually runs - real E(3)-equivariant diffusion generation,
> real OpenMM physics, real STRING/OmniPath/KEGG/PubMed network calls, real
> RDKit chemistry - but no generative/scoring checkpoint here is trained on
> real binding data, and several named tools (Boltz-1, GNINA, DiffSBDD,
> AlphaFold3, FEP+) are documented, unimplemented extension points rather
> than vendored weights this project doesn't have the license/compute to
> ship. **Read [`docs/whitepaper.md`](docs/whitepaper.md) before treating any
> generated molecule, score, or mechanistic hypothesis as validated.** This
> policy - and a meaningful share of the generation/scoring/ADMET engine
> itself - is inherited from [PharmaForge](https://github.com/pharmaforge/pharmaforge),
> vendored here as [`engines/pharmaforge_core`](engines/pharmaforge_core).

## Why BioForge, not another single-model SBDD script

| | A typical SBDD script | BioForge |
|---|---|---|
| Architecture | One notebook/model, run locally | Nine independently deployable FastAPI microservices + a LangGraph orchestrator, Docker/Kubernetes-ready |
| Generation | One diffusion model, sampled once | An ensemble interface (real E(3)-equivariant diffusion today; DiffSBDD/DiffDock-L/latent-diffusion as documented adapter slots) selected via genuine Pareto-optimal multi-objective ranking over affinity, selectivity, synthetic accessibility, and ADMET |
| Structure/pocket | Fpocket or nothing | PDB/UniProt/AlphaFold DB intake + an ensemble of a bounded LIGSITE-style classical grid scan **and** a geometric-deep-learning (GNN) druggability re-ranker |
| Scoring | A single docking number | Differentiable neural scoring + real OpenMM equilibration + gradient-based pose refinement + genuine Monte Carlo dropout uncertainty on every prediction |
| Retrieval | None, or a static CSV | Live PubChem/ChEMBL similarity search, ECFP4/Tanimoto, a real (untrained) equivariant embedding, and RDKit pharmacophore-feature matching |
| **Mechanism** | **None** | **A perturbation graph over the real human PPI network (STRING/OmniPath), network-propagated downstream pathway effects (real KEGG data), and a confidence-scored, literature-anchored causal chain - BioForge's signature feature, see below** |
| Reporting | A printed score | Clinical Translation Score, uncertainty flags on every field, a polypharmacology network view, synthetic-tractability + patent-landscape notes |
| Provenance | None | Every stage logs params/metrics/artifacts to MLflow (or a local JSONL trail if MLflow isn't running) |

## The killer feature: Causal Mechanistic Interpretability & Systems Pharmacology

For every candidate, BioForge doesn't stop at "here's a docking score." It:

1. **Builds a perturbation graph** mapping the primary target and predicted
   off-targets onto the real human PPI network (a live STRING API query,
   falling back to OmniPath).
2. **Propagates the perturbation** through that graph via a real
   random-walk-with-restart (personalized PageRank) - the standard
   systems-biology technique for estimating a perturbation's downstream
   reach - then runs **differential pathway analysis** against real KEGG
   pathway memberships for the most-affected genes.
3. **Pulls supporting literature** via a live PubMed E-utilities search for
   each step of the resulting chain.
4. **Writes a natural-language mechanistic hypothesis** - an offline,
   deterministic template writer by default (no API key required), or a
   real Anthropic API call (true RAG over the computed chain, not free-form
   generation) when configured.
5. **Emits a confidence-scored causal chain**, e.g.:

   ```
   Compound X inhibits PTGS2 [confidence: 1.00; supporting literature: 5 papers]
     -> downregulates Arachidonic acid metabolism [confidence: 0.71; supporting literature: 5 papers]
     -> downregulates Serotonergic synapse [confidence: 0.44; supporting literature: 2 papers]
   ```

This is genuinely computed from live biological databases every run - not a
template with the target's name substituted in. See
[`src/bioforge/core/mechanism.py`](src/bioforge/core/mechanism.py) for the
full implementation and its honesty caveats (network propagation is a real
but weaker proxy for a rigorous do-calculus causal estimate; a `dowhy`-based
refinement is a documented extension point, not implemented here).

## Architecture

```
                          ┌─────────────┐
                          │   gateway   │  API-first REST entry point, OpenAPI
                          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │ orchestrator│  LangGraph StateGraph
                          └──────┬──────┘
        ┌─────────┬─────────┬───┴────┬──────────┬───────────┬────────┐
        ▼         ▼         ▼        ▼          ▼           ▼        ▼
   structure   generate   refine  retrieval    admet    mechanism  report
   (intake +   (ensemble  (OpenMM  (PubChem/   (ADMET/  (STRING/   (synthesis,
   pocket ML)  diffusion  + pose   ChEMBL/     hERG/    OmniPath/  patent,
               + Pareto   refine)  ZINC/       off-     KEGG/      clinical
               ranking)            DrugBank)   target)  PubMed)    score)
```

Every stage is its own FastAPI service (`src/bioforge/services/*`) sharing a
common science layer (`src/bioforge/core/*`) built on the vendored
`engines/pharmaforge_core` engine, a Pydantic schema contract
(`src/bioforge/common/schemas.py`), and a Redis/Celery-ready messaging layer
that every service degrades gracefully without. See
[`docs/architecture.md`](docs/architecture.md) for the full module map.

## Quickstart

```bash
git clone <repo-url> && cd BioForge
pip install -e ./engines/pharmaforge_core
pip install -e ".[dev,frontend]"

# Run the full pipeline against a small bundled protein, in-process, no servers needed
bioforge run --pdb-id 1CRN --name crambin --n-candidates 5 --max-atoms 20

# Fast preview: skip refinement/retrieval/mechanism, print one line per candidate
bioforge run --pdb-id 1CRN --name crambin --n-candidates 2 --max-atoms 5 --n-timesteps 5 \
  --no-refine --no-retrieval --no-mechanism --summary

# Export the receptor + 3D-embedded candidate ligands + a PyMOL script
bioforge run --pdb-id 1CRN --name crambin --n-candidates 2 --dump-dir out/pymol
pymol out/pymol/load_session.pml

# ...or launch the dashboard
streamlit run frontend/streamlit_app/app.py

# ...or run it as real microservices
docker compose up --build
open http://localhost:8501   # dashboard
open http://localhost:8000/docs  # gateway OpenAPI
```

Programmatic use via the Python SDK:

```python
from bioforge.sdk import BioForgeClient
from bioforge.common.schemas import PipelineRunRequest, TargetIntake, TargetSource

client = BioForgeClient(base_url="http://localhost:8000")
result = client.run_pipeline(
    PipelineRunRequest(target=TargetIntake(source=TargetSource.PDB_ID, value="6LU7", name="Mpro"))
)
for report in result.reports:
    print(report.candidate.smiles, report.mechanism.predicted_moa if report.mechanism else None)
```

## Repository layout

```
engines/pharmaforge_core/   vendored generation/scoring/selectivity/ADMET engine (real, tested)
src/bioforge/
├── common/                  shared schemas, config, provenance, messaging
├── core/                    the real science: structure, generation, refinement,
│                             retrieval, ADMET, mechanism, reporting
├── services/                one FastAPI app per pipeline stage
├── orchestrator/            LangGraph StateGraph wiring every stage together
├── sdk/                     Python client
└── cli.py                   `bioforge run` / `bioforge serve <service>`
frontend/streamlit_app/      dashboard
k8s/                         Kubernetes manifests (Kustomize base)
tests/                       unit (no network) + integration (real APIs) suites
docs/                        architecture, whitepaper, arXiv preprint skeleton, Sphinx docs
notebooks/                   tutorial notebooks
benchmarks/                  DUD-E / LIT-PCBA / custom-holdout harness
```

## Testing

```bash
pytest tests/unit -q -m "not integration"        # fast, no network, CI default
pytest tests/integration -q -m integration       # real RCSB/PubChem/STRING/KEGG/PubMed calls
pytest tests -q --cov=src/bioforge --cov-report=term-missing
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md) - module map, data flow, design rationale.
- [`docs/whitepaper.md`](docs/whitepaper.md) - what's real, what's a documented stub, and why, stage by stage.
- [`docs/paper/preprint_skeleton.md`](docs/paper/preprint_skeleton.md) - an arXiv preprint skeleton for a BioForge system paper.
- [`k8s/README.md`](k8s/README.md) - deploying to Kubernetes.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) - dev setup and contribution guidelines.

## License

MIT - see [`LICENSE`](LICENSE). The vendored engine at `engines/pharmaforge_core`
carries its own MIT license and attribution; see
[`engines/pharmaforge_core/UPSTREAM_README.md`](engines/pharmaforge_core/UPSTREAM_README.md).
