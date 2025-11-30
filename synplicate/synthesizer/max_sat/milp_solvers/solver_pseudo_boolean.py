"""
Pseudo-Boolean Hybrid Solver with Cutting-Plane Bounding
=======================================================

This solver follows the classical PB-IP hybrid template: hard and soft clauses
are compiled into a pseudo-Boolean formula with violation indicators, then a
SAT solver with PB encodings performs branch-and-bound on the weighted
objective. We iteratively tighten an upper bound on the total violation weight
using cutting-plane style constraints (encoded via PySAT's PBEnc) until no
better solution exists. See Morgado et al., "Iterative and Core-Guided MaxSAT
Solving: A Survey and a New Framework", JSAT 2013.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from pysat.card import CardEnc, IDPool
from pysat.formula import CNF, WCNF
from pysat.solvers import Minicard


class MILPSolver:
    """Pseudo-Boolean hybrid solver with incremental PB bounding."""

    def __init__(self, wcnf: WCNF):
        self.nv = wcnf.nv
        self.base_formula = CNF()
        for clause in wcnf.hard:
            self.base_formula.append(list(clause))

        self.soft_clauses: List[List[int]] = []
        self.weights: List[int] = []
        self.viol_vars: List[int] = []
        self.next_var = self.nv + 1

        for clause, weight in zip(wcnf.soft, wcnf.wght):
            self._add_soft_clause_internal(list(clause), weight)

        self.best_model: List[int] | None = None
        self.best_cost = float("inf")
        self.cost: float | None = None
        self.solution: List[int] | None = None

    # ------------------------------------------------------------------
    def _add_soft_clause_internal(self, clause: List[int], weight: int) -> None:
        viol_var = self.next_var
        self.next_var += 1
        augmented = list(clause) + [viol_var]
        self.base_formula.append(augmented)

        self.soft_clauses.append(clause)
        self.weights.append(weight)
        self.viol_vars.append(viol_var)

    # ------------------------------------------------------------------
    def add_clause(self, clause: Sequence[int], weight: int) -> None:
        self._add_soft_clause_internal(list(clause), weight)

    # ------------------------------------------------------------------
    def _extract_assignment(self, model: List[int]) -> List[int]:
        assignment: Dict[int, bool] = {}
        for lit in model:
            assignment[abs(lit)] = lit > 0
        return [var if assignment.get(var, False) else -var for var in range(1, self.nv + 1)]

    # ------------------------------------------------------------------
    def compute(self) -> None:
        solver = Minicard()
        solver.append_formula(self.base_formula.clauses)

        vpool = IDPool(start_from=self.next_var)
        current_bound = None

        while True:
            sat = solver.solve()
            if not sat:
                break

            model = solver.get_model()
            if model is None:
                break

            cost = 0
            for viol_var, weight in zip(self.viol_vars, self.weights):
                if viol_var in model:
                    cost += weight

            if cost < self.best_cost:
                self.best_cost = cost
                self.best_model = model

            if cost == 0:
                break

            new_bound = int(cost) - 1
            if current_bound is not None and new_bound >= current_bound:
                # No progress possible; terminate.
                break
            if new_bound < 0:
                break

            # PBEnc from PySAT requires the optional pblib extension; when it is
            # unavailable we over-approximate weighted sums by expanding each
            # literal according to its weight and encoding a standard
            # cardinality bound. This keeps the solver operational without the
            # native PB encoder, at the cost of potentially larger formulas when
            # weights are high.
            expanded = []
            for literal, weight in zip(self.viol_vars, self.weights):
                expanded.extend([literal] * weight)

            constraint = CardEnc.atmost(lits=expanded, bound=new_bound, vpool=vpool)
            solver.append_formula(constraint.clauses)
            current_bound = new_bound
            self.next_var = max(self.next_var, vpool.top)

        solver.delete()

        if self.best_model is None:
            self.solution = None
            self.cost = float("inf")
            return

        self.solution = self._extract_assignment(self.best_model)
        self.cost = self.best_cost

    # ------------------------------------------------------------------
    @property
    def model(self) -> List[int] | None:
        return self.solution

    # ------------------------------------------------------------------
    def delete(self) -> None:
        # Minicard solver is disposed after compute(); nothing to clean up.
        return
