"""
Quadratic MILP Encoding with McCormick Linearization
====================================================

This encoding starts from a quadratic view of MaxSAT where each clause
contributes a satisfaction indicator y_j = 1 - \prod_{l \in C_j} (1 - l(x)). The
quadratic terms are linearized using McCormick envelopes, resulting in a pure
0-1 MILP. The approach is inspired by penalty-based formulations for SAT/MaxSAT
(e.g., Hooker & Ottosson, "Logic-Based Benders Decomposition", Math. Prog. 2003)
and standard bilinear linearization techniques (McCormick, Math. Prog. 1976). By
explicitly modeling the product of unsatisfied-literal indicators, the relaxation
remains relatively tight for small clauses and exposes structure that modern MIP
solvers can exploit.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import gurobipy as gp
from gurobipy import GRB
from pysat.formula import WCNF


class MILPSolver:
    """Quadratic-style MILP solver using McCormick linearization."""

    def __init__(self, wcnf: WCNF):
        self.gurobi_model = gp.Model("quadratic_mccormick_maxsat")
        self.gurobi_model.setParam("OutputFlag", 1)

        self.variables: Dict[int, gp.Var] = {}
        self.clause_vars: Dict[int, gp.Var] = {}
        self.total_weight = 0.0
        self.cost: float | None = None
        self.solution: List[int] | None = None

        self._build_model_from_wcnf(wcnf)

    # ------------------------------------------------------------------
    def _get_var(self, var_id: int) -> gp.Var:
        if var_id not in self.variables:
            self.variables[var_id] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{var_id}")
        return self.variables[var_id]

    # ------------------------------------------------------------------
    def _create_unsat_indicator(self, clause_idx: int, pos: int, literal: int) -> gp.Var:
        base_var = self._get_var(abs(literal))
        indicator = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"u_{clause_idx}_{pos}")
        if literal > 0:
            self.gurobi_model.addConstr(indicator + base_var == 1, name=f"unsat_{clause_idx}_{pos}_pos")
        else:
            self.gurobi_model.addConstr(indicator - base_var == 0, name=f"unsat_{clause_idx}_{pos}_neg")
        return indicator

    # ------------------------------------------------------------------
    def _linearize_product(self, clause_idx: int, factors: Sequence[gp.Var]) -> gp.Var:
        if not factors:
            raise ValueError("Clause must contain at least one literal.")
        product = factors[0]
        for step, factor in enumerate(factors[1:], start=1):
            prod_var = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"p_{clause_idx}_{step}")
            self.gurobi_model.addConstr(prod_var <= product, name=f"mc_{clause_idx}_{step}_1")
            self.gurobi_model.addConstr(prod_var <= factor, name=f"mc_{clause_idx}_{step}_2")
            self.gurobi_model.addConstr(prod_var >= product + factor - 1, name=f"mc_{clause_idx}_{step}_3")
            product = prod_var
        return product

    # ------------------------------------------------------------------
    def _build_model_from_wcnf(self, wcnf: WCNF) -> None:
        for var_id in range(1, wcnf.nv + 1):
            self._get_var(var_id)

        obj = gp.LinExpr()
        clause_counter = 0

        for clause in wcnf.hard:
            y_var = self._encode_clause(clause_counter, clause, hard=True)
            self.gurobi_model.addConstr(y_var == 1, name=f"hard_{clause_counter}")
            clause_counter += 1

        for clause, weight in zip(wcnf.soft, wcnf.wght):
            y_var = self._encode_clause(clause_counter, clause, hard=False)
            obj += weight * y_var
            self.total_weight += weight
            clause_counter += 1

        self.gurobi_model.setObjective(obj, GRB.MAXIMIZE)
        self.gurobi_model.update()

    # ------------------------------------------------------------------
    def _encode_clause(self, clause_idx: int, clause: Sequence[int], hard: bool) -> gp.Var:
        unsat_indicators = [
            self._create_unsat_indicator(clause_idx, pos, lit)
            for pos, lit in enumerate(clause)
        ]
        if len(unsat_indicators) == 1:
            product = unsat_indicators[0]
        else:
            product = self._linearize_product(clause_idx, unsat_indicators)

        y_var = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"y_{clause_idx}")
        self.clause_vars[clause_idx] = y_var

        self.gurobi_model.addConstr(y_var + product == 1, name=f"clause_{clause_idx}_link")
        if hard:
            self.gurobi_model.addConstr(y_var == 1, name=f"clause_{clause_idx}_hard")
        return y_var

    # ------------------------------------------------------------------
    def add_clause(self, clause: Sequence[int], weight: int) -> None:
        idx = len(self.clause_vars)
        y_var = self._encode_clause(idx, clause, hard=False)

        current_obj = self.gurobi_model.getObjective()
        self.gurobi_model.setObjective(current_obj + weight * y_var, GRB.MAXIMIZE)
        self.total_weight += weight
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

        satisfied_weight = self.gurobi_model.ObjVal
        self.cost = self.total_weight - satisfied_weight
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