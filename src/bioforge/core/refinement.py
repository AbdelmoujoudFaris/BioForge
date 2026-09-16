"""Physics-Based Refinement.

Three real, working pieces stand in for the heavier tools named in the
platform spec, each documented honestly about what it is and isn't:

  * `equilibrate_ligand` - a genuine OpenMM minimization + short Langevin
    dynamics run, but against a *generic elastic-network force field* built
    directly from RDKit-perceived bonds/angles and per-element Lennard-Jones
    parameters + Gasteiger charges - not a validated all-atom force field
    (GAFF2/OpenFF). Wiring a real small-molecule force field needs
    `openff-toolkit`'s AM1-BCC charge pipeline (external `ambertools`/
    `openeye` dependency this environment doesn't have reliably installed);
    see `LearnedForceField` below for that extension point.
  * `refine_pose` - gradient-based local pose optimization using the
    vendored, differentiable `ScoringStack` from `pharmaforge_core`
    (`differentiable_score` is literally gradient-w.r.t.-coordinates, per
    its docstring) as a lightweight substitute for a GNINA/DiffDock pose
    search. `GninaPoseAdapter`/`DiffDockPoseAdapter` are the documented,
    unimplemented extension points for the real tools.
  * `pose_uncertainty` - real Monte Carlo dropout: the scoring stack's
    dropout layers are kept active at inference time and sampled N times,
    reporting the mean/std of predicted affinity as an uncertainty estimate
    (Gal & Ghahramani, 2016) - not from an ensemble of independently trained
    models (there's only one, untrained, checkpoint here), but a genuine,
    cheap uncertainty signal over the same weights.

Metadynamics / FEP+ is not implemented: both need a fully parameterized,
validated force field and a much longer simulation budget than a request
handler can spend; `FEPAdapter` documents the extension point.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from pharmaforge_core.models.scoring import ScoringStack
from pharmaforge_core.utils.chem import vdw_radius

logger = logging.getLogger(__name__)

_ELEMENT_MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
                  "P": 30.974, "S": 32.06, "Cl": 35.45, "Br": 79.904, "I": 126.90}
_KCAL_TO_KJ = 4.184


@dataclass
class EquilibrationResult:
    method: str
    n_atoms: int
    initial_potential_energy_kj_mol: float
    final_potential_energy_kj_mol: float
    rmsd_to_input_angstrom: float
    converged: bool


class LearnedForceField:
    """Extension point for a validated small-molecule force field (GAFF2 via
    `openmmforcefields`, or SMIRNOFF via `openff-toolkit`). Not wired here:
    reliable AM1-BCC partial-charge assignment needs an external `ambertools`/
    `openeye` toolchain this deployment doesn't ship; `equilibrate_ligand`
    uses a generic elastic-network potential instead (see module docstring).
    """

    def parameterize(self, mol: Chem.Mol):  # pragma: no cover
        raise NotImplementedError("Wire a real GAFF2/OpenFF parameterization pipeline here.")


def _build_generic_openmm_system(mol: Chem.Mol):
    """Elastic-network-style OpenMM system: harmonic bonds/angles pinned near
    the input geometry (keeps the topology roughly intact during relaxation)
    plus a generic nonbonded term (element-derived vdW radii + RDKit
    Gasteiger charges) so sterically clashing atoms still repel each other.
    """
    import openmm
    from openmm import unit

    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    system = openmm.System()

    AllChem.ComputeGasteigerCharges(mol)
    for atom in mol.GetAtoms():
        elem = atom.GetSymbol()
        system.addParticle(_ELEMENT_MASS.get(elem, 12.0) * unit.amu)

    bonded_pairs = set()
    bond_force = openmm.HarmonicBondForce()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonded_pairs.add((min(i, j), max(i, j)))
        p_i, p_j = conf.GetAtomPosition(i), conf.GetAtomPosition(j)
        length_nm = p_i.Distance(p_j) / 10.0
        bond_force.addBond(i, j, length_nm * unit.nanometer, 250000.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    system.addForce(bond_force)

    angle_force = openmm.HarmonicAngleForce()
    for atom in mol.GetAtoms():
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        for a in range(len(neighbors)):
            for b in range(a + 1, len(neighbors)):
                i, j, k = neighbors[a], atom.GetIdx(), neighbors[b]
                p_i, p_j, p_k = conf.GetAtomPosition(i), conf.GetAtomPosition(j), conf.GetAtomPosition(k)
                v1 = np.array([p_i.x - p_j.x, p_i.y - p_j.y, p_i.z - p_j.z])
                v2 = np.array([p_k.x - p_j.x, p_k.y - p_j.y, p_k.z - p_j.z])
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
                theta = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                angle_force.addAngle(i, j, k, theta * unit.radian, 400.0 * unit.kilojoule_per_mole / unit.radian**2)
    system.addForce(angle_force)

    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for atom in mol.GetAtoms():
        elem = atom.GetSymbol()
        sigma_nm = 2.0 * vdw_radius(elem) / 10.0 / (2 ** (1 / 6))
        try:
            charge = float(atom.GetProp("_GasteigerCharge"))
            if not np.isfinite(charge):
                charge = 0.0
        except (KeyError, ValueError):
            charge = 0.0
        nonbonded.addParticle(charge, sigma_nm * unit.nanometer, 0.4 * unit.kilojoule_per_mole)
    for i, j in bonded_pairs:
        nonbonded.addException(i, j, 0.0, 0.1, 0.0)
    system.addForce(nonbonded)

    return system, n


def equilibrate_ligand(mol: Chem.Mol, n_md_steps: int = 200, temperature_k: float = 300.0) -> EquilibrationResult:
    """Energy-minimize then run a short Langevin equilibration in vacuum.
    Returns before/after potential energy and heavy-atom RMSD to the input
    pose so callers can see how much the generic force field moved the
    ligand (a large RMSD is a signal the raw diffusion-sampled geometry had
    real steric/geometric problems worth flagging).
    """
    import openmm
    from openmm import unit

    mol = Chem.Mol(mol)
    if mol.GetNumConformers() == 0:
        raise ValueError("equilibrate_ligand requires a mol with an embedded 3D conformer")

    system, n = _build_generic_openmm_system(mol)
    integrator = openmm.LangevinMiddleIntegrator(
        temperature_k * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond
    )
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, integrator, platform)

    conf = mol.GetConformer()
    initial_positions = np.array([list(conf.GetAtomPosition(i)) for i in range(n)]) / 10.0  # A -> nm
    context.setPositions(initial_positions * unit.nanometer)

    initial_state = context.getState(getEnergy=True)
    initial_pe = initial_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=200)
    context.setVelocitiesToTemperature(temperature_k * unit.kelvin)
    integrator.step(n_md_steps)

    final_state = context.getState(getEnergy=True, getPositions=True)
    final_pe = final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    final_positions = np.array(final_state.getPositions().value_in_unit(unit.nanometer)) * 10.0  # nm -> A

    rmsd = float(np.sqrt(np.mean(np.sum((final_positions - initial_positions * 10.0) ** 2, axis=1))))

    return EquilibrationResult(
        method="openmm_generic_elastic_network_v1",
        n_atoms=n,
        initial_potential_energy_kj_mol=float(initial_pe),
        final_potential_energy_kj_mol=float(final_pe),
        rmsd_to_input_angstrom=rmsd,
        converged=bool(final_pe <= initial_pe + 1e-3),
    )


# ---------------------------------------------------------------------------
# Pose prediction / refinement (GNINA / DiffDock substitute)
# ---------------------------------------------------------------------------


class GninaPoseAdapter:
    """Extension point for a real GNINA docking run. Not implemented: GNINA
    needs a compiled CUDA/CPU binary this environment doesn't ship.
    """

    def dock(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Wire a real `gnina` binary invocation here.")


class DiffDockPoseAdapter:
    """Extension point for a real DiffDock inference run. Not implemented:
    needs the published checkpoint and a GPU.
    """

    def dock(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Wire a real DiffDock checkpoint + inference call here.")


@dataclass
class PoseRefinementResult:
    method: str
    n_steps: int
    initial_score: float
    final_score: float
    coords: np.ndarray


def refine_pose(
    scoring_stack: ScoringStack,
    pocket: dict,
    ligand_atom_type_idx: torch.Tensor,
    ligand_physchem: torch.Tensor,
    ligand_coords: torch.Tensor,
    n_steps: int = 30,
    lr: float = 0.05,
) -> PoseRefinementResult:
    """Local pose refinement by gradient ascent on the differentiable
    scoring stack's predicted score, w.r.t. ligand coordinates - a
    lightweight, real substitute for a GNINA/DiffDock search (see
    `GninaPoseAdapter`/`DiffDockPoseAdapter` for the real-tool extension
    points). Improves *local* pose quality only; it cannot escape the basin
    the diffusion model's sampled pose started in.
    """
    scoring_stack.eval()  # disable dropout for a deterministic gradient signal
    coords = ligand_coords.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([coords], lr=lr)

    with torch.no_grad():
        initial_score = float(
            scoring_stack.differentiable_score(pocket, ligand_atom_type_idx, ligand_physchem, ligand_coords).item()
        )

    for _ in range(n_steps):
        optimizer.zero_grad()
        score = scoring_stack.differentiable_score(pocket, ligand_atom_type_idx, ligand_physchem, coords)
        loss = -score
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_score = float(
            scoring_stack.differentiable_score(pocket, ligand_atom_type_idx, ligand_physchem, coords).item()
        )

    return PoseRefinementResult(
        method="differentiable_scoring_gradient_refinement",
        n_steps=n_steps,
        initial_score=initial_score,
        final_score=final_score,
        coords=coords.detach().numpy(),
    )


# ---------------------------------------------------------------------------
# Uncertainty quantification (MC dropout)
# ---------------------------------------------------------------------------


@dataclass
class UncertaintyEstimate:
    method: str
    mean: float
    std: float
    n_samples: int
    flagged_low_confidence: bool


def _enable_dropout_only(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_uncertainty(
    scoring_stack: ScoringStack,
    pocket: dict,
    ligand_atom_type_idx: torch.Tensor,
    ligand_physchem: torch.Tensor,
    ligand_coords: torch.Tensor,
    n_samples: int = 20,
    std_flag_threshold: float = 0.75,
) -> UncertaintyEstimate:
    """Monte Carlo dropout (Gal & Ghahramani, 2016): keep dropout active at
    inference and sample the affinity head N times. Std across samples is
    the uncertainty estimate - genuine epistemic-uncertainty signal from a
    single checkpoint, not from an ensemble of independently trained models.
    """
    scoring_stack.eval()
    _enable_dropout_only(scoring_stack)
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = scoring_stack(pocket, ligand_atom_type_idx, ligand_physchem, ligand_coords)
            samples.append(float(out["binding_affinity"].item()))
    scoring_stack.eval()

    mean, std = float(np.mean(samples)), float(np.std(samples))
    return UncertaintyEstimate(
        method="mc_dropout_ensemble_disagreement",
        mean=mean,
        std=std,
        n_samples=n_samples,
        flagged_low_confidence=std > std_flag_threshold,
    )


class FEPAdapter:
    """Extension point for a real free-energy-perturbation / metadynamics
    workflow (e.g. OpenFE, PLUMED-driven OpenMM metadynamics). Not
    implemented: needs a validated force field (see `LearnedForceField`) and
    a simulation budget (ns-scale, many replicas) far beyond what a request
    handler can spend; production deployments should run this as an
    offline/batch job against the shortlisted top candidates only.
    """

    def run(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Wire a real FEP/metadynamics workflow here; see class docstring.")
