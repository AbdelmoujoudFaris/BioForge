"""`bioforge` command-line entry point.

  bioforge run --pdb-id 6LU7 --name Mpro --n-candidates 5
  bioforge serve gateway              # any of: gateway structure generate refine retrieval admet mechanism report orchestrator
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _dump_for_pymol(result, dump_dir: str) -> None:
    """Write a receptor PDB, per-candidate 3D ligand PDBs, and a ready-to-run
    PyMOL script into `dump_dir` so the run's output can be loaded directly."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    out = Path(dump_dir)
    out.mkdir(parents=True, exist_ok=True)

    pml_lines: list[str] = []

    receptor_src = result.structure.pdb_path
    receptor_dst = out / "receptor.pdb"
    if receptor_src and Path(receptor_src).is_file():
        shutil.copyfile(receptor_src, receptor_dst)
        pml_lines.append(f'load {receptor_dst.name}, receptor')
    else:
        print(f"warning: no receptor pdb_path on structure record; skipping receptor copy", file=sys.stderr)

    selected = result.pocket.selected_pocket_id
    center = next((c.center for c in result.pocket.candidates if c.pocket_id == selected), None)
    if center is not None:
        x, y, z = center
        pml_lines.append(f'pseudoatom pocket_center, pos=[{x}, {y}, {z}]')
        pml_lines.append('show spheres, pocket_center')
        pml_lines.append('color red, pocket_center')

    n_written = 0
    for report in result.reports:
        c = report.candidate
        if not c.valid or not c.smiles:
            continue
        mol = Chem.MolFromSmiles(c.smiles)
        if mol is None:
            print(f"warning: rdkit could not parse SMILES for {c.candidate_id}: {c.smiles!r}", file=sys.stderr)
            continue
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            print(f"warning: rdkit 3D embedding failed for {c.candidate_id}", file=sys.stderr)
            continue
        AllChem.MMFFOptimizeMolecule(mol)
        short_id = c.candidate_id.split("-")[0]
        ligand_path = out / f"ligand_{short_id}.pdb"
        Chem.MolToPDBFile(mol, str(ligand_path))
        pml_lines.append(f'load {ligand_path.name}, ligand_{short_id}')
        n_written += 1

    pml_lines.append('bg_color white')
    pml_lines.append('show cartoon, receptor')
    pml_lines.append('hide lines, receptor')
    pml_lines.append('zoom')

    script_path = out / "load_session.pml"
    script_path.write_text("\n".join(pml_lines) + "\n")

    print(f"Wrote receptor + {n_written} ligand(s) + PyMOL script to: {out.resolve()}")
    print(f"Open with: pymol {script_path.resolve()}")


def _cmd_run(args: argparse.Namespace) -> None:
    from bioforge.common.schemas import PipelineRunRequest, TargetIntake, TargetSource
    from bioforge.orchestrator.graph import run_pipeline

    req = PipelineRunRequest(
        target=TargetIntake(source=TargetSource.PDB_ID, value=args.pdb_id, name=args.name, ligand_resname=args.ligand),
        n_candidates=args.n_candidates,
        max_atoms=args.max_atoms,
        n_timesteps=args.n_timesteps,
        run_refinement=not args.no_refine,
        run_retrieval=not args.no_retrieval,
        run_mechanism=not args.no_mechanism,
        seed=args.seed,
    )
    result = run_pipeline(req)
    if args.summary:
        elapsed_s = (result.finished_at - result.started_at).total_seconds()
        print(f"run_id={result.run_id} target={result.target.name} elapsed={elapsed_s:.1f}s "
              f"candidates={len(result.reports)}/{req.n_candidates}")
        for report in result.reports:
            c = report.candidate
            print(f"  {c.candidate_id}  valid={c.valid}  pareto={c.pareto_optimal}  "
                  f"binding={c.scores.binding_affinity_kcal_mol}  qed={c.scores.qed}  "
                  f"sa={c.scores.sa_score}  smiles={c.smiles}")
    else:
        print(result.model_dump_json(indent=2))

    if args.dump_dir:
        _dump_for_pymol(result, args.dump_dir)


def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    ports = {
        "gateway": 8000, "structure": 8001, "generate": 8002, "refine": 8003,
        "retrieval": 8004, "admet": 8005, "mechanism": 8006, "report": 8007, "orchestrator": 8008,
    }
    if args.service not in ports:
        print(f"Unknown service {args.service!r}; choose from {sorted(ports)}", file=sys.stderr)
        raise SystemExit(2)
    uvicorn.run(f"bioforge.services.{args.service}.main:app", host="0.0.0.0", port=args.port or ports[args.service])


def main() -> None:
    parser = argparse.ArgumentParser(prog="bioforge")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the full pipeline in-process against a PDB ID")
    run_p.add_argument("--pdb-id", required=True)
    run_p.add_argument("--name", default="target")
    run_p.add_argument("--ligand", default=None, help="Reference co-crystallized ligand residue name")
    run_p.add_argument("--n-candidates", type=int, default=5)
    run_p.add_argument("--max-atoms", type=int, default=30)
    run_p.add_argument("--n-timesteps", type=int, default=50)
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--no-refine", action="store_true")
    run_p.add_argument("--no-retrieval", action="store_true")
    run_p.add_argument("--no-mechanism", action="store_true")
    run_p.add_argument("--summary", action="store_true", help="Print a short per-candidate summary instead of the full JSON result")
    run_p.add_argument("--dump-dir", default=None,
                        help="Write receptor.pdb, per-candidate 3D ligand PDBs, and a load_session.pml into this directory for PyMOL")
    run_p.set_defaults(func=_cmd_run)

    serve_p = sub.add_parser("serve", help="Run one BioForge microservice with uvicorn")
    serve_p.add_argument("service")
    serve_p.add_argument("--port", type=int, default=None)
    serve_p.set_defaults(func=_cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
