"""Selectivity Profiler: PharmaForge's signature differentiator.

Given one primary target and an arbitrary panel of anti-targets (off-target
proteins whose binding you want to *avoid*), this module drives molecule
generation toward high primary-target affinity while explicitly penalizing
affinity against every anti-target, and reports a per-molecule
**selectivity index** SI = Kd_offtarget / Kd_target (higher is better;
SI >> 1 means the molecule strongly prefers the primary target).

Two complementary mechanisms are provided:

1. `constrained_guidance_score` - a scalarized penalty term pluggable
   straight into `LigandDiffusionModel.p_sample_loop`'s `guidance_fn`, for
   single-run selectivity-aware *sampling* (fast, differentiable).

2. `SelectivityMOOProblem` + `run_multi_objective_search` - a proper
   multi-objective optimization loop (NSGA-III by default, MOEA/D as an
   alternative) over a *population* of already-generated candidates,
   producing a Pareto front that trades off primary affinity against every
   anti-target simultaneously instead of a single scalar weighting. This is
   what the dashboard's "Pareto front" browser (ui/dashboard.py) consumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from pharmaforge_core.config import SelectivityConfig
from pharmaforge_core.models.scoring import MultiTargetScoringEnsemble

_KCAL_PER_MOL_TO_RT = 0.593  # RT at 298K in kcal/mol, for ΔG -> Kd conversion


def delta_g_to_kd(delta_g_kcal_mol: float) -> float:
    """Kd (M) = exp(ΔG / RT). ΔG more negative -> tighter binding -> smaller Kd."""
    return float(np.exp(delta_g_kcal_mol / _KCAL_PER_MOL_TO_RT))


def selectivity_index(delta_g_target: float, delta_g_offtarget: float) -> float:
    """SI = Kd_offtarget / Kd_target. >> 1 means selective for the primary target."""
    kd_target = delta_g_to_kd(delta_g_target)
    kd_offtarget = delta_g_to_kd(delta_g_offtarget)
    return kd_offtarget / max(kd_target, 1e-12)


@dataclass
class SelectivityProfile:
    target_delta_g: float
    offtarget_delta_g: dict[str, float]
    selectivity_index: dict[str, float] = field(init=False)

    def __post_init__(self):
        self.selectivity_index = {
            name: selectivity_index(self.target_delta_g, dg) for name, dg in self.offtarget_delta_g.items()
        }

    @property
    def worst_case_si(self) -> float:
        return min(self.selectivity_index.values()) if self.selectivity_index else float("inf")


class SelectivityProfiler:
    """Wraps a `MultiTargetScoringEnsemble` to score one molecule against a
    primary target + anti-target panel, and to provide the differentiable
    penalty term used during constrained guided-diffusion sampling.
    """

    def __init__(self, ensemble: MultiTargetScoringEnsemble, cfg: SelectivityConfig):
        self.ensemble = ensemble
        self.cfg = cfg

    def profile(
        self,
        target_name: str,
        anti_target_names: list[str],
        pockets: dict[str, dict],
        ligand_atom_type_idx: Tensor,
        ligand_physchem: Tensor,
        ligand_coords: Tensor,
    ) -> SelectivityProfile:
        target_score = self.ensemble.score(
            target_name, pockets[target_name], ligand_atom_type_idx, ligand_physchem, ligand_coords
        )
        target_dg = -target_score.item()  # differentiable_score negates ΔG; undo for reporting
        offtarget_dg = {}
        for name in anti_target_names:
            s = self.ensemble.score(name, pockets[name], ligand_atom_type_idx, ligand_physchem, ligand_coords)
            offtarget_dg[name] = -s.item()
        return SelectivityProfile(target_dg, offtarget_dg)

    def constrained_guidance_score(
        self,
        target_name: str,
        anti_target_names: list[str],
        pockets: dict[str, dict],
        ligand_atom_type_idx: Tensor,
        ligand_physchem: Tensor,
        ligand_coords: Tensor,
    ) -> Tensor:
        """Differentiable scalar: maximize target affinity, penalize every
        anti-target affinity. Used as `guidance_fn` during diffusion sampling
        so selectivity is enforced *during* generation, not filtered after.
        """
        target_score = self.ensemble.score(
            target_name, pockets[target_name], ligand_atom_type_idx, ligand_physchem, ligand_coords
        )
        penalty = torch.zeros((), device=ligand_coords.device)
        for name in anti_target_names:
            offtarget_score = self.ensemble.score(
                name, pockets[name], ligand_atom_type_idx, ligand_physchem, ligand_coords
            )
            # offtarget_score is "higher = tighter off-target binding" (same
            # convention as target_score), so penalize it directly.
            penalty = penalty + torch.relu(offtarget_score)
        return target_score - self.cfg.anti_target_penalty_weight * penalty


# --------------------------------------------------------------------------
# Population-level multi-objective search (NSGA-III / MOEA-D via pymoo)
# --------------------------------------------------------------------------


def _lazy_import_pymoo():
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions

    return NSGA3, MOEAD, Problem, minimize, get_reference_directions


def build_selectivity_problem(
    candidate_scores: np.ndarray,
    Problem,
):
    """candidate_scores: (n_candidates, 1 + n_anti_targets) matrix of raw
    delta-G-like scores per already-generated candidate, column 0 = primary
    target, columns 1..k = anti-targets. This runs the multi-objective search
    over a *discrete, pre-scored candidate pool* (typical for post-hoc
    Pareto-front selection in the dashboard) rather than over continuous
    generation parameters.
    """
    n_candidates, n_objectives = candidate_scores.shape

    class SelectivitySelection(Problem):
        """Objectives: minimize primary ΔG (tighter binding) and maximize
        every anti-target ΔG (weaker binding) simultaneously, encoded as a
        binary selection mask over the candidate pool with a fixed budget.
        """

        def __init__(self, budget: int):
            super().__init__(n_var=n_candidates, n_obj=n_objectives, n_constr=1, xl=0, xu=1, vtype=bool)
            self.budget = budget

        def _evaluate(self, x, out, *args, **kwargs):
            x_bin = x > 0.5
            counts = x_bin.sum(axis=1).astype(float)
            counts[counts == 0] = 1.0
            primary = (x_bin * candidate_scores[:, 0]).sum(axis=1) / counts
            objs = [primary]
            for k in range(1, n_objectives):
                offtarget = (x_bin * candidate_scores[:, k]).sum(axis=1) / counts
                objs.append(-offtarget)  # maximize off-target ΔG == minimize -ΔG
            out["F"] = np.column_stack(objs)
            out["G"] = np.abs(x_bin.sum(axis=1) - self.budget) - 0.5  # ~= budget selected

    return SelectivitySelection


def _n_das_dennis_dirs(n_objectives: int, n_partitions: int) -> int:
    from math import comb

    return comb(n_partitions + n_objectives - 1, n_objectives - 1)


def _auto_partitions(n_objectives: int, population_size: int, max_partitions: int = 12) -> int:
    """Largest n_partitions whose das-dennis reference-direction count still
    fits inside the population (NSGA3 warns, and can fail to converge, when
    pop_size < len(ref_dirs)); falls back to 1 rather than 0 partitions.
    """
    best = 1
    for p in range(1, max_partitions + 1):
        if _n_das_dennis_dirs(n_objectives, p) <= population_size:
            best = p
        else:
            break
    return best


def run_multi_objective_search(
    candidate_scores: np.ndarray,
    cfg: SelectivityConfig,
    budget: int | None = None,
):
    """Returns the Pareto-optimal subset of candidate indices trading off
    primary-target affinity against every anti-target's affinity.

    Raises `RuntimeError` (rather than silently returning an empty front) if
    the search converges without a single feasible individual - this can
    happen with a very small population/generation budget, since the
    "exactly `budget` candidates selected" constraint is combinatorial.
    Increase `cfg.population_size`/`cfg.n_generations` if this triggers.
    """
    NSGA3, MOEAD, Problem, minimize, get_reference_directions = _lazy_import_pymoo()
    n_candidates, n_objectives = candidate_scores.shape
    budget = budget or max(1, n_candidates // 10)
    problem_cls = build_selectivity_problem(candidate_scores, Problem)
    problem = problem_cls(budget=budget)

    n_partitions = _auto_partitions(n_objectives, cfg.population_size)
    ref_dirs = get_reference_directions("das-dennis", n_objectives, n_partitions=n_partitions)
    if cfg.method == "moead":
        algorithm = MOEAD(ref_dirs=ref_dirs, n_neighbors=15, prob_neighbor_mating=0.7)
    else:
        algorithm = NSGA3(pop_size=cfg.population_size, ref_dirs=ref_dirs)

    result = minimize(problem, algorithm, ("n_gen", cfg.n_generations), seed=1, verbose=False)
    if result.F is None:
        raise RuntimeError(
            f"Multi-objective search found no feasible selection of exactly {budget} candidates "
            f"within {cfg.n_generations} generations (population_size={cfg.population_size}). "
            "Increase population_size/n_generations, or lower the budget."
        )
    return result


def pareto_front_indices(candidate_scores: np.ndarray) -> np.ndarray:
    """Direct non-dominated sort over an already-scored candidate pool
    (column 0 = primary target ΔG to minimize, columns 1..k = anti-target
    ΔG to maximize i.e. minimize their negation) - what the dashboard's
    Pareto-front browser actually needs: "which of these N generated
    molecules are Pareto-optimal", with no combinatorial subset-selection
    or population-size tuning required.
    """
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    objectives = candidate_scores.copy()
    objectives[:, 1:] = -objectives[:, 1:]
    fronts = NonDominatedSorting().do(objectives, only_non_dominated_front=True)
    return np.asarray(fronts)
