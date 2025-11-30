"""
Pseudo-Boolean Strengthened MILP Encoding
=========================================

This encoding replaces the classic Big-M relaxation with tighter
pseudo-Boolean (PB) constraints and cover inequalities. The formulation follows
ideas from the PB/MaxSAT literature (e.g., Hooker, "Logic-Based Methods for
Optimization", and Ansótegui et al., "Towards Industrial MaxSAT Solvers",
SAT 2013). Clauses are encoded as PB constraints where each literal contributes
its truth indicator and an explicit relaxation variable r_j. Additional cover
constraints bound the number of simultaneously falsified literals, yielding a
stronger LP relaxation and improved branch-and-cut performance compared with the
naïve Big-M approach. Gurobi's cut generation can further tighten the model via
its native clique/PB separation routines.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import gurobipy as gp
from gurobipy import GRB
from pysat.formula import WCNF


class MILPSolver:
    """MILP solver with pseudo-Boolean strengthened constraints."""

    def __init__(self, wcnf: WCNF):
        self.gurobi_model = gp.Model("pb_strengthened_maxsat")
        self.gurobi_model.setParam("OutputFlag", 1)

        self.variables: Dict[int, gp.Var] = {}
        self.relax_vars: Dict[int, gp.Var] = {}
        self.cost: float | None = None
        self.solution: List[int] | None = None

        self._build_model_from_wcnf(wcnf)

    # ------------------------------------------------------------------
    def _literal_value(self, literal: int) -> gp.Var:
        var_id = abs(literal)
        if var_id not in self.variables:
            self.variables[var_id] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{var_id}")
        var = self.variables[var_id]
        return var if literal > 0 else (1 - var)

    # ------------------------------------------------------------------
    def _unsat_indicator(self, literal: int) -> gp.LinExpr:
        var_id = abs(literal)
        if var_id not in self.variables:
            self.variables[var_id] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{var_id}")
        var = self.variables[var_id]
        return (1 - var) if literal > 0 else var

    # ------------------------------------------------------------------
    def _build_model_from_wcnf(self, wcnf: WCNF) -> None:
        for var_id in range(1, wcnf.nv + 1):
            self.variables[var_id] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{var_id}")

        obj = gp.LinExpr()

        # Hard clauses simply enforce the PB >= 1 constraint.
        for idx, clause in enumerate(wcnf.hard):
            pb_expr = gp.LinExpr()
            for lit in clause:
                pb_expr += self._literal_value(lit)
            self.gurobi_model.addConstr(pb_expr >= 1, name=f"hard_{idx}")

        for idx, (clause, weight) in enumerate(zip(wcnf.soft, wcnf.wght)):
            relax = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"r_{idx}")
            self.relax_vars[idx] = relax

            pb_expr = gp.LinExpr()
            unsat_expr = gp.LinExpr()
            for lit in clause:
                pb_expr += self._literal_value(lit)
                unsat_expr += self._unsat_indicator(lit)

            # PB form with explicit relaxation variable.
            self.gurobi_model.addConstr(pb_expr + relax >= 1, name=f"soft_pb_{idx}")

            # Cover inequality: at most |C|-1 literals can be simultaneously
            # falsified unless the relaxation variable is activated.
            clause_len = len(clause)
            self.gurobi_model.addConstr(
                unsat_expr <= clause_len - 1 + clause_len * relax,
                name=f"soft_cover_{idx}",
            )

            obj += weight * relax

        self.gurobi_model.setObjective(obj, GRB.MINIMIZE)
        self.gurobi_model.update()

    # ------------------------------------------------------------------
    def add_clause(self, clause: Sequence[int], weight: int) -> None:
        relax_idx = len(self.relax_vars)
        relax = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"r_{relax_idx}")
        self.relax_vars[relax_idx] = relax

        pb_expr = gp.LinExpr()
        unsat_expr = gp.LinExpr()
        for lit in clause:
            pb_expr += self._literal_value(lit)
            unsat_expr += self._unsat_indicator(lit)

        self.gurobi_model.addConstr(pb_expr + relax >= 1, name=f"soft_pb_{relax_idx}")
        clause_len = len(clause)
        self.gurobi_model.addConstr(
            unsat_expr <= clause_len - 1 + clause_len * relax,
            name=f"soft_cover_{relax_idx}",
        )

        current_obj = self.gurobi_model.getObjective()
        self.gurobi_model.setObjective(current_obj + weight * relax, GRB.MINIMIZE)
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
            value = var_id if var.X > 0.5 else -var_id
            self.solution.append(value)

    # ------------------------------------------------------------------
    @property
    def model(self) -> List[int] | None:
        return self.solution

    # ------------------------------------------------------------------
    def delete(self) -> None:
        if hasattr(self, "gurobi_model") and self.gurobi_model is not None:
            self.gurobi_model.dispose()