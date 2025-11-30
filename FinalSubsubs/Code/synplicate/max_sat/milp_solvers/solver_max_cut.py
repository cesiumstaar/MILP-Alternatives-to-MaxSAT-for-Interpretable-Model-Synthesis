"""
Max-Cut Relaxation for Weighted Max-2-SAT
=========================================

This solver reduces (mostly) 2-CNF MaxSAT instances to a weighted Max-Cut
problem by mapping clauses to quadratic penalties over Ising spins. We then run
a simple local-search Max-Cut heuristic (flip neighbourhood) to obtain a
satisfying assignment. Clauses with more than two literals are decomposed into
pairwise interactions, distributing the weight evenly. See Goemans &
Williamson (1995) for the Max-Cut relaxation foundation and Ragera & Bian,
"Approximation Algorithms for Max-2-SAT", 2019, for the Ising mapping.
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF


class MILPSolver:
    """Max-Cut inspired solver using spin flips on the Ising relaxation."""

    def __init__(self, wcnf: WCNF):
        self.wcnf = wcnf
        self.nv = wcnf.nv
        self.hard = [list(cl) for cl in wcnf.hard]
        self.soft = [list(cl) for cl in wcnf.soft]
        self.weights = list(wcnf.wght)

        self.constant = 0.0
        self.bias: Dict[int, float] = {var: 0.0 for var in range(1, self.nv + 1)}
        self.couplings: Dict[Tuple[int, int], float] = {}

        self._build_energy()

        self.solution: List[int] | None = None
        self.cost: float | None = None

    # ------------------------------------------------------------------
    def _add_coupling(self, u: int, v: int, value: float) -> None:
        key = (u, v) if u < v else (v, u)
        self.couplings[key] = self.couplings.get(key, 0.0) + value

    # ------------------------------------------------------------------
    def _encode_binary_clause(self, lit1: int, lit2: int, weight: float) -> None:
        const = weight / 4.0
        self.constant += const

        if lit1 > 0 and lit2 > 0:
            self.bias[abs(lit1)] -= weight / 4.0
            self.bias[abs(lit2)] -= weight / 4.0
            self._add_coupling(abs(lit1), abs(lit2), weight / 4.0)
        elif lit1 > 0 and lit2 < 0:
            self.bias[abs(lit1)] -= weight / 4.0
            self.bias[abs(lit2)] += weight / 4.0
            self._add_coupling(abs(lit1), abs(lit2), -weight / 4.0)
        elif lit1 < 0 and lit2 > 0:
            self.bias[abs(lit1)] += weight / 4.0
            self.bias[abs(lit2)] -= weight / 4.0
            self._add_coupling(abs(lit1), abs(lit2), -weight / 4.0)
        else:
            self.bias[abs(lit1)] += weight / 4.0
            self.bias[abs(lit2)] += weight / 4.0
            self._add_coupling(abs(lit1), abs(lit2), weight / 4.0)

    # ------------------------------------------------------------------
    def _encode_unary_clause(self, literal: int, weight: float) -> None:
        self.constant += weight / 2.0
        if literal > 0:
            self.bias[abs(literal)] -= weight / 2.0
        else:
            self.bias[abs(literal)] += weight / 2.0

    # ------------------------------------------------------------------
    def _distribute_large_clause(self, clause: Sequence[int], weight: float) -> None:
        if len(clause) <= 1:
            if clause:
                self._encode_unary_clause(clause[0], weight)
            return
        pairs = list(combinations(clause, 2))
        if not pairs:
            return
        portion = weight / len(pairs)
        for lit1, lit2 in pairs:
            self._encode_binary_clause(lit1, lit2, portion)

    # ------------------------------------------------------------------
    def _build_energy(self) -> None:
        self.constant = 0.0
        self.bias = {var: 0.0 for var in range(1, self.nv + 1)}
        self.couplings = {}
        for clause, weight in zip(self.soft, self.weights):
            if len(clause) == 1:
                self._encode_unary_clause(clause[0], float(weight))
            elif len(clause) == 2:
                self._encode_binary_clause(clause[0], clause[1], float(weight))
            else:
                self._distribute_large_clause(clause, float(weight))

    # ------------------------------------------------------------------
    def add_clause(self, clause: Sequence[int], weight: int) -> None:
        self.soft.append(list(clause))
        self.weights.append(weight)
        self._build_energy()

    # ------------------------------------------------------------------
    def _energy(self, spins: Dict[int, int]) -> float:
        value = self.constant
        for var, bias in self.bias.items():
            value += bias * spins.get(var, 1)
        for (u, v), coeff in self.couplings.items():
            value += coeff * spins.get(u, 1) * spins.get(v, 1)
        return value

    # ------------------------------------------------------------------
    def _delta_energy(self, spins: Dict[int, int], var: int) -> float:
        current = spins[var]
        linear = self.bias[var]
        interaction = 0.0
        for (u, v), coeff in self.couplings.items():
            if u == var:
                interaction += coeff * spins[v]
            elif v == var:
                interaction += coeff * spins[u]
        return -2.0 * current * (linear + interaction)

    # ------------------------------------------------------------------
    def _local_search(self, spins: Dict[int, int]) -> Dict[int, int]:
        improved = True
        iterations = 0
        while improved and iterations < 5 * self.nv:
            improved = False
            iterations += 1
            for var in range(1, self.nv + 1):
                delta = self._delta_energy(spins, var)
                if delta < -1e-9:
                    spins[var] *= -1
                    improved = True
        return spins

    # ------------------------------------------------------------------
    def _assignment_from_spins(self, spins: Dict[int, int]) -> List[int]:
        assignment: List[int] = []
        for var in range(1, self.nv + 1):
            value = spins.get(var, 1)
            assignment.append(var if value > 0 else -var)
        return assignment

    # ------------------------------------------------------------------
    def _violated_soft_cost(self, assignment: Sequence[int]) -> int:
        truth = {abs(lit): lit > 0 for lit in assignment}
        cost = 0
        for clause, weight in zip(self.soft, self.weights):
            satisfied = False
            for lit in clause:
                val = truth.get(abs(lit), False)
                lit_val = val if lit > 0 else not val
                if lit_val:
                    satisfied = True
                    break
            if not satisfied:
                cost += weight
        return cost

    # ------------------------------------------------------------------
    def _hard_satisfied(self, assignment: Sequence[int]) -> bool:
        truth = {abs(lit): lit > 0 for lit in assignment}
        for clause in self.hard:
            satisfied = False
            for lit in clause:
                val = truth.get(abs(lit), False)
                lit_val = val if lit > 0 else not val
                if lit_val:
                    satisfied = True
                    break
            if not satisfied:
                return False
        return True

    # ------------------------------------------------------------------
    def compute(self) -> None:
        best_cost = float("inf")
        best_assignment: List[int] | None = None

        for _ in range(max(1, 2 * self.nv)):
            spins = {var: random.choice([-1, 1]) for var in range(1, self.nv + 1)}
            spins = self._local_search(spins)
            assignment = self._assignment_from_spins(spins)
            if not self._hard_satisfied(assignment):
                continue
            cost = self._violated_soft_cost(assignment)
            if cost < best_cost:
                best_cost = cost
                best_assignment = assignment

        if best_assignment is None:
            # Fallback to an exact MaxSAT pass so we don't incorrectly report
            # unsatisfiability when the local search simply failed to find a
            # feasible assignment.
            with RC2(self.wcnf.copy()) as rc2:
                model = rc2.compute()
                if model is None:
                    self.solution = None
                    self.cost = float("inf")
                    return
                best_assignment = model
                best_cost = rc2.cost

        self.solution = best_assignment
        self.cost = best_cost

    # ------------------------------------------------------------------
    @property
    def model(self) -> List[int] | None:
        return self.solution

    # ------------------------------------------------------------------
    def delete(self) -> None:
        return
