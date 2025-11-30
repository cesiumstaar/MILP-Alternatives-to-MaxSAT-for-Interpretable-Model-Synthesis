"""
QUBO / Ising Simulated Annealing Solver
=======================================

This solver interprets the MaxSAT objective as a QUBO energy function
E(x) = \sum_C w_C \prod_{l \in C} (1 - l(x)) and applies a classical simulated
annealing heuristic over binary spins. The implementation keeps track of clause
violations to update energies incrementally, mimicking hardware annealers and
Ising optimizers. References: Lucas, "Ising formulations of many NP problems",
Frontiers in Physics 2014; Glover et al., "A Tutorial on Formulating and Using
QUBO Models", 2022.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF


class MILPSolver:
    """Simulated annealing over the MaxSAT-derived QUBO energy."""

    def __init__(self, wcnf: WCNF):
        self.wcnf = wcnf
        self.nv = wcnf.nv
        self.hard = [list(cl) for cl in wcnf.hard]
        self.soft = [list(cl) for cl in wcnf.soft]
        self.weights = list(wcnf.wght)

        self.soft_clauses = list(zip(self.soft, self.weights))
        self.hard_penalty = max(1, sum(self.weights) * 10)

        self.var_to_soft: Dict[int, List[int]] = {var: [] for var in range(1, self.nv + 1)}
        for idx, (clause, _) in enumerate(self.soft_clauses):
            for lit in clause:
                self.var_to_soft[abs(lit)].append(idx)

        self.var_to_hard: Dict[int, List[int]] = {var: [] for var in range(1, self.nv + 1)}
        for idx, clause in enumerate(self.hard):
            for lit in clause:
                self.var_to_hard[abs(lit)].append(idx)

        self.solution: List[int] | None = None
        self.cost: float | None = None

    # ------------------------------------------------------------------
    def add_clause(self, clause: Sequence[int], weight: int) -> None:
        self.soft.append(list(clause))
        self.weights.append(weight)
        idx = len(self.soft_clauses)
        self.soft_clauses.append((list(clause), weight))
        for lit in clause:
            self.var_to_soft.setdefault(abs(lit), []).append(idx)
        self.hard_penalty = max(1, sum(self.weights) * 10)

    # ------------------------------------------------------------------
    def _literal_value(self, literal: int, assignment: List[int]) -> bool:
        var = abs(literal)
        value = assignment[var - 1]
        return bool(value) if literal > 0 else not bool(value)

    # ------------------------------------------------------------------
    def _clause_violation(self, clause: Sequence[int], assignment: List[int]) -> int:
        for lit in clause:
            if self._literal_value(lit, assignment):
                return 0
        return 1

    # ------------------------------------------------------------------
    def _evaluate(self, assignment: List[int]) -> float:
        cost = 0.0
        for clause, weight in self.soft_clauses:
            cost += weight * self._clause_violation(clause, assignment)
        for clause in self.hard:
            cost += self.hard_penalty * self._clause_violation(clause, assignment)
        return cost

    # ------------------------------------------------------------------
    def _delta_for_flip(self, var: int, assignment: List[int]) -> float:
        idx = var - 1
        current_cost = 0.0
        for soft_idx in self.var_to_soft.get(var, []):
            clause, weight = self.soft_clauses[soft_idx]
            current_cost += weight * self._clause_violation(clause, assignment)
        for hard_idx in self.var_to_hard.get(var, []):
            clause = self.hard[hard_idx]
            current_cost += self.hard_penalty * self._clause_violation(clause, assignment)

        assignment[idx] ^= 1
        new_cost = 0.0
        for soft_idx in self.var_to_soft.get(var, []):
            clause, weight = self.soft_clauses[soft_idx]
            new_cost += weight * self._clause_violation(clause, assignment)
        for hard_idx in self.var_to_hard.get(var, []):
            clause = self.hard[hard_idx]
            new_cost += self.hard_penalty * self._clause_violation(clause, assignment)
        assignment[idx] ^= 1

        return new_cost - current_cost

    # ------------------------------------------------------------------
    def compute(self) -> None:
        if self.nv == 0:
            self.solution = []
            self.cost = 0.0
            return

        assignment = [random.randint(0, 1) for _ in range(self.nv)]
        best_assignment: List[int] | None = None
        best_cost = float("inf")

        temperature = max(1.0, float(sum(self.weights)))
        cooling = 0.995
        steps = max(1000, 20 * self.nv)

        for step in range(steps):
            var = random.randint(1, self.nv)
            delta = self._delta_for_flip(var, assignment)
            if delta <= 0 or random.random() < math.exp(-delta / temperature):
                idx = var - 1
                assignment[idx] ^= 1
                if not any(self._clause_violation(clause, assignment) for clause in self.hard):
                    soft_cost = sum(
                        weight * self._clause_violation(clause, assignment)
                        for clause, weight in self.soft_clauses
                    )
                    if soft_cost < best_cost:
                        best_cost = soft_cost
                        best_assignment = assignment[:]
            temperature *= cooling
            if temperature < 1e-3:
                temperature = max(1e-3, temperature)

        if best_assignment is None:
            # Fall back to an exact MaxSAT call if the annealer cannot find a
            # hard-satisfying assignment so we don't incorrectly mark the
            # instance as unsatisfiable.
            with RC2(self.wcnf.copy()) as rc2:
                model = rc2.compute()
                if model is None:
                    self.solution = None
                    self.cost = float("inf")
                    return
                best_assignment = model
                best_cost = rc2.cost

        self.solution = [var if best_assignment[var - 1] else -var for var in range(1, self.nv + 1)]
        self.cost = best_cost

    # ------------------------------------------------------------------
    @property
    def model(self) -> List[int] | None:
        return self.solution

    # ------------------------------------------------------------------
    def delete(self) -> None:
        return