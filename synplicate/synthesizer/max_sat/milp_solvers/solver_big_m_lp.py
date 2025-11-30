"""
Big-M Linear Programming Relaxation for MaxSAT
==============================================

This solver mirrors the classic Big-M MILP encoding for partial MaxSAT but
relaxes all Boolean and slack variables to continuous values in [0, 1]. The
resulting LP can be solved extremely quickly and provides a fractional solution
that we round back to a Boolean assignment. This approach is useful as a fast
lower bound or warm-start for combinatorial solvers; see Nemhauser & Wolsey,
"Integer and Combinatorial Optimization" (1988) for a discussion of Big-M and
LP relaxations.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import gurobipy as gp
from gurobipy import GRB
from pysat.formula import WCNF


class MILPSolver:
    """Continuous Big-M relaxation with RC2-style interface."""

    def __init__(self, wcnf: WCNF):
        self.lp_model = gp.Model("big_m_lp_relaxation")
        self.lp_model.setParam("OutputFlag", 0)

        self.variables: Dict[int, gp.Var] = {}
        self.slack_vars: Dict[int, gp.Var] = {}
        self.cost: float | None = None
        self.solution: List[int] | None = None
        self.obj = gp.LinExpr()

        self._nv = wcnf.nv
        self._hard = [list(cl) for cl in wcnf.hard]
        self._soft = [list(cl) for cl in wcnf.soft]
        self._weights = list(wcnf.wght)

        self._build_model(wcnf)

    # ------------------------------------------------------------------
    def _build_model(self, wcnf: WCNF) -> None:
        for var in range(1, wcnf.nv + 1):
            self.variables[var] = self.lp_model.addVar(
                vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"x_{var}"
            )

        for idx, clause in enumerate(self._hard):
            expr = gp.LinExpr()
            for lit in clause:
                expr += self._literal_expr(lit)
            self.lp_model.addConstr(expr >= 1, name=f"hard_{idx}")

        for idx, (clause, weight) in enumerate(zip(self._soft, self._weights)):
            slack = self.lp_model.addVar(
                vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"s_{idx}"
            )
            self.slack_vars[idx] = slack

            expr = gp.LinExpr()
            for lit in clause:
                expr += self._literal_expr(lit)
            self.lp_model.addConstr(expr + slack >= 1, name=f"soft_{idx}")
            self.obj += weight * slack

        self.lp_model.setObjective(self.obj, GRB.MINIMIZE)
        self.lp_model.update()

    # ------------------------------------------------------------------
    def _literal_expr(self, literal: int) -> gp.LinExpr:
        var = self.variables[abs(literal)]
        return var if literal > 0 else (1 - var)

    # ------------------------------------------------------------------
    def add_clause(self, clause: Sequence[int], weight: int) -> None:
        idx = len(self.slack_vars)
        slack = self.lp_model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"s_{idx}")
        self.slack_vars[idx] = slack
        self._soft.append(list(clause))
        self._weights.append(weight)

        expr = gp.LinExpr()
        for lit in clause:
            expr += self._literal_expr(lit)
        self.lp_model.addConstr(expr + slack >= 1, name=f"soft_{idx}")

        self.obj += weight * slack
        self.lp_model.setObjective(self.obj, GRB.MINIMIZE)
        self.lp_model.update()

    # ------------------------------------------------------------------
    def compute(self) -> None:
        try:
            self.lp_model.optimize()
        except gp.GurobiError as exc:
            print(f"Gurobi error during LP relaxation: {exc}")
            self.solution = None
            self.cost = float("inf")
            return

        if self.lp_model.status != GRB.OPTIMAL:
            print("LP relaxation failed to find an optimal solution.")
            self.solution = None
            self.cost = float("inf")
            return

        self.cost = self.lp_model.ObjVal
        rounded: List[int] = []
        for var_id in range(1, self._nv + 1):
            var = self.variables[var_id]
            value = 1 if var.X >= 0.5 else -1
            rounded.append(value * var_id)
        self.solution = rounded

    # ------------------------------------------------------------------
    @property
    def model_values(self) -> List[int] | None:
        return self.solution

    # ------------------------------------------------------------------
    @property
    def model(self) -> List[int] | None:
        return self.solution

    # ------------------------------------------------------------------
    def delete(self) -> None:
        if hasattr(self, "lp_model") and self.lp_model is not None:
            self.lp_model.dispose()
            self.lp_model = None