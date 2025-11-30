"""
SAT Master with MIP Cost Evaluation
==================================

This hybrid approach enumerates satisfying assignments of the hard clauses with
a SAT solver (PySAT's Minisat22) and, for each model, evaluates the weighted
soft-clause objective using a small Big-M style linear program. The SAT solver
provides strong pruning over the Boolean search space while the MIP is
responsible for handling the weighted objective and any future linear side
constraints. This mirrors the "SAT master, MIP slave" pattern discussed by
Heras et al., "An Overview of Algorithms for Max-SAT", Constraints 2008.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import gurobipy as gp
from gurobipy import GRB
from pysat.formula import WCNF
from pysat.solvers import Minisat22


class MILPSolver:
    """SAT-guided enumeration with MIP-backed cost evaluation."""

    def __init__(self, wcnf: WCNF):
        self._nv = wcnf.nv
        self._hard = [list(cl) for cl in wcnf.hard]
        self._soft = [list(cl) for cl in wcnf.soft]
        self._weights = list(wcnf.wght)

        self.sat_solver = Minisat22()
        for clause in self._hard:
            self.sat_solver.add_clause(list(clause))

        self.best_cost = float("inf")
        self.best_assignment: Dict[int, bool] | None = None
        self.solution: List[int] | None = None
        self.cost: float | None = None

    # ------------------------------------------------------------------
    def add_clause(self, clause: Sequence[int], weight: int) -> None:
        self._soft.append(list(clause))
        self._weights.append(weight)

    # ------------------------------------------------------------------
    def _evaluate_assignment(self, assignment: Dict[int, bool]) -> float:
        model = gp.Model("sat_mip_eval")
        model.setParam("OutputFlag", 0)

        decision_vars: Dict[int, gp.Var] = {}
        for var in range(1, self._nv + 1):
            fixed_val = 1.0 if assignment.get(var, False) else 0.0
            decision_vars[var] = model.addVar(
                vtype=GRB.CONTINUOUS, lb=fixed_val, ub=fixed_val, name=f"x_{var}"
            )

        slack_vars: List[gp.Var] = []
        obj = gp.LinExpr()
        for idx, (clause, weight) in enumerate(zip(self._soft, self._weights)):
            slack = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"s_{idx}")
            slack_vars.append(slack)

            expr = gp.LinExpr()
            for lit in clause:
                lit_var = decision_vars[abs(lit)]
                expr += lit_var if lit > 0 else (1 - lit_var)
            model.addConstr(expr + slack >= 1, name=f"soft_{idx}")
            obj += weight * slack

        model.setObjective(obj, GRB.MINIMIZE)
        try:
            model.optimize()
        except gp.GurobiError as exc:
            print(f"Gurobi error during SAT+MIP evaluation: {exc}")
            model.dispose()
            return float("inf")

        if model.status != GRB.OPTIMAL:
            model.dispose()
            return float("inf")

        value = model.ObjVal
        model.dispose()
        return value

    # ------------------------------------------------------------------
    def _assignment_from_model(self, model: List[int]) -> Dict[int, bool]:
        assignment: Dict[int, bool] = {}
        for lit in model:
            var = abs(lit)
            if var <= self._nv:
                assignment[var] = lit > 0
        return assignment

    # ------------------------------------------------------------------
    def compute(self) -> None:
        while self.sat_solver.solve():
            sat_model = self.sat_solver.get_model()
            if sat_model is None:
                break
            assignment = self._assignment_from_model(sat_model)
            cost = self._evaluate_assignment(assignment)

            if cost < self.best_cost:
                self.best_cost = cost
                self.best_assignment = assignment

            block: List[int] = []
            for var in range(1, self._nv + 1):
                value = assignment.get(var, False)
                block.append(-var if value else var)
            self.sat_solver.add_clause(block)

        if self.best_assignment is None:
            self.solution = None
            self.cost = float("inf")
            return

        self.cost = self.best_cost
        self.solution = [var if val else -var for var, val in sorted(self.best_assignment.items())]

    # ------------------------------------------------------------------
    @property
    def model(self) -> List[int] | None:
        return self.solution

    # ------------------------------------------------------------------
    def delete(self) -> None:
        if hasattr(self, "sat_solver") and self.sat_solver is not None:
            self.sat_solver.delete()
            self.sat_solver = None