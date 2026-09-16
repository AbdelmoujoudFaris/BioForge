# Contributing to BioForge

## Dev setup

```bash
git clone <repo-url> && cd BioForge
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ./engines/pharmaforge_core
pip install -e ".[dev,scale,tracking,physics,frontend]"
pre-commit install
```

## Before opening a PR

```bash
ruff check src tests
ruff format src tests
pytest tests/unit -q -m "not integration"
```

Integration tests (`pytest tests/integration -q -m integration`) hit real
RCSB/PubChem/STRING/OmniPath/KEGG/PubMed APIs - run them locally when you
touch `core/structure_intel.py`, `core/retrieval.py`, or `core/mechanism.py`,
but don't worry if they're flaky in CI (upstream services, not this repo).

## Where things live

- **`engines/pharmaforge_core/`** - the vendored generation/scoring/
  selectivity/ADMET engine. Treat it as a near-upstream dependency: prefer
  fixing bugs here over routing around them in `src/bioforge/core`, and keep
  its own test suite (`engines/pharmaforge_core/tests/`) green.
- **`src/bioforge/core/`** - real science logic, framework-free (no FastAPI
  imports). Every function here should be directly unit-testable.
- **`src/bioforge/services/*/main.py`** - thin FastAPI wrappers over `core/`.
  Keep business logic out of these files; they're intentionally excluded
  from the coverage requirement (`pyproject.toml`'s `[tool.coverage.run]`)
  because they're not supposed to have branches worth testing beyond what
  `tests/unit/test_services_api.py` already covers.
- **`src/bioforge/orchestrator/graph.py`** - the LangGraph pipeline. Node
  bodies call `core/` in-process; that's a deployment choice (see the module
  docstring), not an architectural rule - a distributed deployment can swap
  a node body for an HTTP call via `bioforge.sdk` without touching the graph
  topology.

## Honesty policy

This codebase's guiding rule, inherited from PharmaForge: **a real, working,
simpler implementation beats a fake one that name-drops a bigger tool.**
Concretely:

- If you can't wire real weights/API access for something the platform spec
  names (Boltz-1, GNINA, FEP+, DrugBank, ...), add a documented extension
  point that raises `NotImplementedError` with a clear pointer to what's
  missing (see `StructurePredictorAdapter`, `GninaPoseAdapter`,
  `LearnedADMET` for the pattern) - never fabricate a plausible-looking
  number.
- If a result depends on untrained/randomly-initialized weights, say so in
  the docstring and, where user-facing, in the API/dashboard output.
- Prefer a smaller, real, tested implementation (e.g. `equilibrate_ligand`'s
  generic elastic-network force field instead of a half-wired GAFF2
  pipeline) over a bigger one you can't verify actually runs.
- Update `docs/whitepaper.md` when you add or remove one of these caveats.

## Adding a new pipeline stage / adapter

1. Add the real logic to `src/bioforge/core/<module>.py` with unit tests in
   `tests/unit/`.
2. Add/extend the Pydantic schema in `src/bioforge/common/schemas.py`.
3. Wire a FastAPI endpoint in the relevant `src/bioforge/services/*/main.py`
   (or a new service directory, mirroring the existing ones, with its own
   `Dockerfile` port assignment - see `docker-compose.yml` and
   `k8s/base/deployments.yaml`).
4. Wire it into `src/bioforge/orchestrator/graph.py` if it belongs in the
   default pipeline run.
5. Add a `bioforge.sdk.client.BioForgeClient` method if it's meant for
   programmatic/SDK use.

## Commit / PR conventions

- Keep PRs scoped to one stage/module where possible.
- Explain *why*, not just *what*, in the PR description - especially for any
  change to a documented honesty caveat.
- New external API dependencies: note whether it's free/keyless (like
  PubChem/STRING/KEGG/PubMed) or needs a license (like DrugBank), and add
  the graceful-fallback path for the latter.
