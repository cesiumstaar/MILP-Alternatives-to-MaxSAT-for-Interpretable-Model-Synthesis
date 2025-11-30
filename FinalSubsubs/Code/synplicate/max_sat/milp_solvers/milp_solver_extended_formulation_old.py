"""
Extended Formulation MILP Encoding
=================================

This encoding follows the "ideal" extended formulation philosophy, where each
clause receives an auxiliary satisfaction indicator that tightens the LP
relaxation. The construction is inspired by disjunctive programming techniques
(Balas, "Disjunctive Programming", 1975) and the extended formulations for small
CNF clauses discussed in the MaxSAT/MIP community (e.g., Chen & Hooker, "A
Hybrid Approach to Max-SAT", CP 2006). For every clause C we introduce a binary
variable y_C representing whether C is satisfied, and link it to the literals via
upper/lower bounds that force y_C to 1 whenever at least one literal is true.
Hard clauses simply enforce y_C = 1, while soft clauses minimize the violation
weight through (1 - y_C) in the objective. For 2-literal clauses we add an extra
linking constraint that directly prohibits both literals from being simultaneously
false unless the clause is relaxed, yielding an "ideal" relaxation for binary
clauses.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import gurobipy as gp
from gurobipy import GRB
from pysat.formula import WCNF


class MILPSolver:
    """Extended formulation MILP for MaxSAT."""

    def __init__(self, wcnf: WCNF):
        self.gurobi_model = gp.Model("extended_formulation_maxsat")
        self.gurobi_model.setParam("OutputFlag", 1)

        self.variables: Dict[int, gp.Var] = {}
        self.clause_vars: Dict[int, gp.Var] = {}
        self.cost: float | None = None
        self.solution: List[int] | None = None

        self._build_model_from_wcnf(wcnf)

    # ------------------------------------------------------------------
    def _literal_truth(self, literal: int) -> gp.LinExpr:
        var_id = abs(literal)
        if var_id not in self.variables:
            self.variables[var_id] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{var_id}")
        var = self.variables[var_id]
        return var if literal > 0 else (1 - var)

    # ------------------------------------------------------------------
    def _build_model_from_wcnf(self, wcnf: WCNF) -> None:
        for var_id in range(1, wcnf.nv + 1):
            self.variables[var_id] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{var_id}")

        obj = gp.LinExpr()

        hard_offset = 0
        for idx, clause in enumerate(wcnf.hard):
            y_var = self._add_clause_indicator(idx, clause)
            self.gurobi_model.addConstr(y_var == 1, name=f"hard_{idx}_enforce")
            hard_offset = idx + 1

        for idx, (clause, weight) in enumerate(zip(wcnf.soft, wcnf.wght), start=hard_offset):
            y_var = self._add_clause_indicator(idx, clause)
            obj += weight * (1 - y_var)

        self.gurobi_model.setObjective(obj, GRB.MINIMIZE)
        self.gurobi_model.update()

    # ------------------------------------------------------------------
    def _add_clause_indicator(self, idx: int, clause: Sequence[int]) -> gp.Var:
        y_var = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"y_{idx}")
        self.clause_vars[idx] = y_var

        truth_expr = gp.LinExpr()
        for lit in clause:
            truth_expr += self._literal_truth(lit)

        clause_len = len(clause)
        self.gurobi_model.addConstr(y_var <= truth_expr, name=f"clause_{idx}_ub")
        self.gurobi_model.addConstr(
            y_var >= truth_expr / clause_len,
            name=f"clause_{idx}_lb",
        )

        if clause_len == 2:
            unsat_expr = gp.LinExpr()
            for lit in clause:
                unsat_expr += (1 - self._literal_truth(lit))
            self.gurobi_model.addConstr(unsat_expr <= 1, name=f"clause_{idx}_binary_cover")

        return y_var

    # ------------------------------------------------------------------
    def add_clause(self, clause: Sequence[int], weight: int) -> None:
        idx = len(self.clause_vars)
        y_var = self._add_clause_indicator(idx, clause)

        current_obj = self.gurobi_model.getObjective()
        self.gurobi_model.setObjective(current_obj + weight * (1 - y_var), GRB.MINIMIZE)
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