"""
Big-M MILP Encoding (Baseline)
==============================

This file implements the classical Big-M formulation for MaxSAT (see e.g.,
Davies & Bacchus, "Solving MAXSAT by Solvers", SAT 2011) where each soft clause
receives a violation binary variable v_j with coefficient M = 1. The constraint
\sum_{l \in C_j} l(x) + v_j >= 1 ensures the clause is satisfied whenever
v_j = 0; setting v_j = 1 allows the clause to be violated at a cost equal to its
weight. Although the Big-M model is easy to derive, it usually produces weak LP
relaxations and serves here as a baseline for more advanced encodings.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import gurobipy as gp
from gurobipy import GRB
from pysat.formula import WCNF


class MILPSolver:
    """Baseline Big-M MILP solver mirroring the RC2-style interface."""

    def __init__(self, wcnf: WCNF):
        self.gurobi_model = gp.Model("big_m_maxsat")
        self.gurobi_model.setParam("OutputFlag", 1)

        self.variables: Dict[int, gp.Var] = {}
        self.slack_vars: Dict[int, gp.Var] = {}
        self.cost: float | None = None
        self.solution: List[int] | None = None

        self._build_model_from_wcnf(wcnf)

    # ------------------------------------------------------------------
    def _build_model_from_wcnf(self, wcnf: WCNF) -> None:
        for var_id in range(1, wcnf.nv + 1):
            self.variables[var_id] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{var_id}")

        obj = gp.LinExpr()

        for idx, clause in enumerate(wcnf.hard):
            expr = gp.LinExpr()
            for literal in clause:
                expr += self._literal_value(literal)
            self.gurobi_model.addConstr(expr >= 1, name=f"hard_{idx}")

        for idx, (clause, weight) in enumerate(zip(wcnf.soft, wcnf.wght)):
            slack = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"v_{idx}")
            self.slack_vars[idx] = slack

            expr = gp.LinExpr()
            for literal in clause:
                expr += self._literal_value(literal)

            self.gurobi_model.addConstr(expr + slack >= 1, name=f"soft_{idx}")
            obj += weight * slack

        self.gurobi_model.setObjective(obj, GRB.MINIMIZE)
        self.gurobi_model.update()

    # ------------------------------------------------------------------
    def _literal_value(self, literal: int) -> gp.LinExpr:
        var_id = abs(literal)
        if var_id not in self.variables:
            self.variables[var_id] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{var_id}")
        var = self.variables[var_id]
        return var if literal > 0 else (1 - var)

    # ------------------------------------------------------------------
    def add_clause(self, clause: Sequence[int], weight: int) -> None:
        idx = len(self.slack_vars)
        slack = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"v_{idx}")
        self.slack_vars[idx] = slack

        expr = gp.LinExpr()
        for literal in clause:
            expr += self._literal_value(literal)

        self.gurobi_model.addConstr(expr + slack >= 1, name=f"soft_{idx}")

        current_obj = self.gurobi_model.getObjective()
        self.gurobi_model.setObjective(current_obj + weight * slack, GRB.MINIMIZE)
        self.gurobi_model.update()

    # ------------------------------------------------------------------
    def compute(self) -> None:
        try:
            self.gurobi_model.optimize()
        except gp.GurobiError as exc:
            print(f"Gurobi error during optimization: {exc}")
            self.solution = None
            self.cost = float("inf")
            return

        if self.gurobi_model.status != GRB.OPTIMAL:
            print("Gurobi solver status is not optimal.")
            self.solution = None
            self.cost = float("inf")
            return

        self.cost = self.gurobi_model.ObjVal
        self.solution = []
        for var_id in range(1, len(self.variables) + 1):
            var = self.variables[var_id]
            self.solution.append(var_id if var.X > 0.5 else -var_id)

    # ------------------------------------------------------------------
    @property
    def model(self) -> List[int] | None:
        return self.solution

    # ------------------------------------------------------------------
    def delete(self) -> None:
        if hasattr(self, "gurobi_model") and self.gurobi_model is not None:
            self.gurobi_model.dispose()
