"""
Constraint Programming Inspired Max-CSP Solver
=============================================

This solver treats the MaxSAT instance as a binary Max-CSP: each variable has a
{0,1} domain, hard clauses become hard constraints, and soft clauses accrue
weights when violated. We implement a depth-first branch-and-bound search with
forward checking similar to MAC (Maintaining Arc Consistency), pruning any
partial assignment that violates a hard clause and bounding by the accumulated
weight of violated soft clauses. See Rossi, van Beek, & Walsh, "Handbook of
Constraint Programming" (2006) for background on Max-CSP algorithms.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from pysat.formula import WCNF


class MILPSolver:
    """Max-CSP style solver using branch-and-bound with forward checking."""

    def __init__(self, wcnf: WCNF):
        self.nv = wcnf.nv
        self.hard = [list(cl) for cl in wcnf.hard]
        self.soft = [list(cl) for cl in wcnf.soft]
        self.weights = list(wcnf.wght)

        self.best_cost = float("inf")
        self.best_assignment: Dict[int, bool] | None = None
        self.solution: List[int] | None = None
        self.cost: float | None = None

        self.var_order = self._build_variable_order()

    # ------------------------------------------------------------------
    def _build_variable_order(self) -> List[int]:
        activity = {var: 0 for var in range(1, self.nv + 1)}
        for clause, weight in zip(self.soft, self.weights):
            for lit in clause:
                activity[abs(lit)] += weight
        for clause in self.hard:
            for lit in clause:
                activity[abs(lit)] += 1
        return [var for var, _ in sorted(activity.items(), key=lambda item: -item[1])]

    # ------------------------------------------------------------------
    def add_clause(self, clause: Sequence[int], weight: int) -> None:
        self.soft.append(list(clause))
        self.weights.append(weight)
        self.var_order = self._build_variable_order()

    # ------------------------------------------------------------------
    def _literal_value(self, literal: int, assignment: Dict[int, bool]) -> bool | None:
        var = abs(literal)
        if var not in assignment:
            return None
        value = assignment[var]
        return value if literal > 0 else not value

    # ------------------------------------------------------------------
    def _clause_status(self, clause: Sequence[int], assignment: Dict[int, bool]) -> bool | None:
        undecided = False
        for lit in clause:
            value = self._literal_value(lit, assignment)
            if value is True:
                return True
            if value is None:
                undecided = True
        return None if undecided else False

    # ------------------------------------------------------------------
    def _violated_soft_cost(self, assignment: Dict[int, bool]) -> int:
        cost = 0
        for clause, weight in zip(self.soft, self.weights):
            status = self._clause_status(clause, assignment)
            if status is False:
                cost += weight
        return cost

    # ------------------------------------------------------------------
    def _hard_consistent(self, assignment: Dict[int, bool]) -> bool:
        for clause in self.hard:
            status = self._clause_status(clause, assignment)
            if status is False:
                return False
        return True

    # ------------------------------------------------------------------
    def _search(self, index: int, assignment: Dict[int, bool]) -> None:
        current_cost = self._violated_soft_cost(assignment)
        if current_cost >= self.best_cost:
            return
        if not self._hard_consistent(assignment):
            return
        if len(assignment) == self.nv:
            self.best_cost = current_cost
            self.best_assignment = dict(assignment)
            return

        # Choose the next variable not yet assigned based on activity ordering.
        for var in self.var_order:
            if var not in assignment:
                next_var = var
                break
        else:
            return

        for value in (True, False):
            assignment[next_var] = value
            self._search(index + 1, assignment)
            del assignment[next_var]

    # ------------------------------------------------------------------
    def compute(self) -> None:
        self._search(0, {})
        if self.best_assignment is None:
            self.solution = None
            self.cost = float("inf")
            return

        self.solution = [var if self.best_assignment.get(var, False) else -var for var in range(1, self.nv + 1)]
        self.cost = self.best_cost

    # ------------------------------------------------------------------
    @property
    def model(self) -> List[int] | None:
        return self.solution

    # ------------------------------------------------------------------
    def delete(self) -> None:
        return