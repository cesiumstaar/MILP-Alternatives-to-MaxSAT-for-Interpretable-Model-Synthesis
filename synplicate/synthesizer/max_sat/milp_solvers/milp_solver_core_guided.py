"""
Core-Guided MILP Encoding (UNSAT-Core / Hitting-Set MILP)
========================================================

This solver implements a core-guided optimization strategy inspired by
state-of-the-art MaxSAT solvers such as MSCG and Open-WBO (see Morgado et al.,
"Core-Guided MaxSAT with Soft Cardinality Constraints", SAT 2013). The idea is
 to start with all relaxation variables fixed to zero and iteratively relax only
those soft clauses that appear in infeasible cores, enforcing a minimal hitting
set of cores via additional constraints. The approach mimics the classical
unsatisfiable-core guided loop:

1. Build an initial MILP model that enforces every soft clause without
   relaxation (r_j = 0).
2. Optimize. If the model is feasible, the solution is optimal.
3. If the model is infeasible, extract an IIS (irreducible infeasible subset)
   from Gurobi, interpret the IIS as an unsatisfiable core, allow the
   corresponding relaxation variables to take value 1, and add a hitting-set
   constraint that requires at least one relaxation variable in the core to be
   activated.
4. Repeat until the MILP becomes feasible.

This emulates UNSAT-core guided MaxSAT within a single MILP model while taking
advantage of Gurobi's incremental solving and IIS computation capabilities.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import gurobipy as gp
from gurobipy import GRB
from pysat.formula import WCNF


class MILPSolver:
    """Core-guided MILP solver that mirrors the :class:`pysat.examples.rc2.RC2` API."""

    def __init__(self, wcnf: WCNF):
        self.gurobi_model = gp.Model("core_guided_maxsat")
        self.gurobi_model.setParam("OutputFlag", 1)

        self.variables: Dict[int, gp.Var] = {}
        self.relax_vars: Dict[int, gp.Var] = {}
        self.soft_constraints: Dict[int, gp.Constr] = {}
        self.hitting_set_id = 0
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
            for lit in clause:
                expr += self._literal_as_var(lit)
            self.gurobi_model.addConstr(expr >= 1, name=f"hard_{idx}")

        for idx, (clause, weight) in enumerate(zip(wcnf.soft, wcnf.wght)):
            relax = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"r_{idx}")
            relax.setAttr(GRB.Attr.UB, 0.0)
            self.relax_vars[idx] = relax

            expr = gp.LinExpr()
            for lit in clause:
                expr += self._literal_as_var(lit)

            constr = self.gurobi_model.addConstr(expr + relax >= 1, name=f"soft_{idx}")
            self.soft_constraints[idx] = constr
            obj += weight * relax

        self.gurobi_model.setObjective(obj, GRB.MINIMIZE)
        self.gurobi_model.update()

    # ------------------------------------------------------------------
    def _literal_as_var(self, literal: int) -> gp.LinExpr:
        var_id = abs(literal)
        if var_id not in self.variables:
            self.variables[var_id] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{var_id}")
        var = self.variables[var_id]
        return var if literal > 0 else (1 - var)

    # ------------------------------------------------------------------
    def add_clause(self, clause: Sequence[int], weight: int) -> None:
        relax_idx = len(self.relax_vars)
        relax = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"r_{relax_idx}")
        relax.setAttr(GRB.Attr.UB, 0.0)
        self.relax_vars[relax_idx] = relax

        expr = gp.LinExpr()
        for lit in clause:
            expr += self._literal_as_var(lit)

        constr = self.gurobi_model.addConstr(expr + relax >= 1, name=f"soft_{relax_idx}")
        self.soft_constraints[relax_idx] = constr

        current_obj = self.gurobi_model.getObjective()
        self.gurobi_model.setObjective(current_obj + weight * relax, GRB.MINIMIZE)
        self.gurobi_model.update()

    # ------------------------------------------------------------------
    def _add_hitting_set_constraint(self, core_indices: Sequence[int]) -> None:
        expr = gp.quicksum(self.relax_vars[idx] for idx in core_indices)
        self.gurobi_model.addConstr(expr >= 1, name=f"hit_{self.hitting_set_id}")
        self.hitting_set_id += 1
        for idx in core_indices:
            relax = self.relax_vars[idx]
            if relax.UB < 1.0:
                relax.setAttr(GRB.Attr.UB, 1.0)
        self.gurobi_model.update()

    # ------------------------------------------------------------------
    def _extract_core_from_iis(self) -> List[int]:
        core: List[int] = []
        for idx, constr in self.soft_constraints.items():
            if constr.IISConstr:
                core.append(idx)
        return core

    # ------------------------------------------------------------------
    def compute(self) -> None:
        while True:
            try:
                self.gurobi_model.optimize()
            except gp.GurobiError as exc:
                print(f"Gurobi error during optimization: {exc}")
                self.solution = None
                self.cost = float("inf")
                return

            status = self.gurobi_model.status
            if status == GRB.OPTIMAL:
                self._extract_solution()
                return
            if status != GRB.INFEASIBLE:
                print("Gurobi solver ended with unexpected status.")
                self.solution = None
                self.cost = float("inf")
                return

            self.gurobi_model.computeIIS()
            core = self._extract_core_from_iis()
            if not core:
                print("Unable to extract unsatisfiable core from IIS.")
                self.solution = None
                self.cost = float("inf")
                return
            self._add_hitting_set_constraint(core)

    # ------------------------------------------------------------------
    def _extract_solution(self) -> None:
        self.cost = self.gurobi_model.ObjVal
        self.solution = []
        for var_id in range(1, len(self.variables) + 1):
            var = self.variables[var_id]
            value = 1 if var.X > 0.5 else -1
            self.solution.append(value * var_id)

    # ------------------------------------------------------------------
    @property
    def model(self) -> List[int] | None:
        return self.solution

    # ------------------------------------------------------------------
    def delete(self) -> None:
        if hasattr(self, "gurobi_model") and self.gurobi_model is not None:
            self.gurobi_model.dispose()