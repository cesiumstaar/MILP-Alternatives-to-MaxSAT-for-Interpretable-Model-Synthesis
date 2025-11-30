"""
Implicit Hitting-Set SAT+MIP Hybrid
==================================

This solver combines a SAT engine for feasibility with a MILP master that
optimizes the selection of relaxed soft clauses. The approach mirrors the
implicit hitting-set paradigm popularized by MaxSAT solvers like MSCG and
MaxHS (see Marques-Silva & Planes, 2011). The master MILP minimizes the weight
of relaxed clauses while the SAT solver checks whether the selected relaxation
set yields a consistent assignment. Unsatisfiable cores from the SAT solver are
translated into hitting-set cuts for the MILP, driving the search toward an
optimal solution.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import gurobipy as gp
from gurobipy import GRB
from pysat.formula import WCNF
from pysat.solvers import Minisat22


class MILPSolver:
    """Implicit hitting-set hybrid with SAT feasibility checks."""

    MAX_ITERATIONS = 512

    def __init__(self, wcnf: WCNF):
        self._nv = wcnf.nv
        self._hard = [list(cl) for cl in wcnf.hard]
        self._soft = [list(cl) for cl in wcnf.soft]
        self._weights = list(wcnf.wght)

        self.master = gp.Model("sat_mip_hitting_set")
        self.master.setParam("OutputFlag", 1)

        self.relax_vars: List[gp.Var] = []
        self.obj = gp.LinExpr()
        for idx, weight in enumerate(self._weights):
            r_var = self.master.addVar(vtype=GRB.BINARY, name=f"r_{idx}")
            self.relax_vars.append(r_var)
            self.obj += weight * r_var
        if self.relax_vars:
            self.master.setObjective(self.obj, GRB.MINIMIZE)
        else:
            self.master.setObjective(gp.LinExpr(0.0))
        self.master.update()

        self.sat_solver = Minisat22()
        for clause in self._hard:
            self.sat_solver.add_clause(list(clause))

        self.assumption_lits: List[int] = []
        self.assumption_to_index: Dict[int, int] = {}
        self.next_aux = self._nv + 1

        for idx, clause in enumerate(self._soft):
            aux = self.next_aux
            self.next_aux += 1
            augmented = list(clause) + [aux]
            self.sat_solver.add_clause(augmented)
            assumption = -aux
            self.assumption_lits.append(assumption)
            self.assumption_to_index[assumption] = idx

        self.seen_cores: Set[Tuple[int, ...]] = set()
        self.cost: float | None = None
        self.solution: List[int] | None = None

    # ------------------------------------------------------------------
    def add_clause(self, clause: Sequence[int], weight: int) -> None:
        idx = len(self.relax_vars)
        r_var = self.master.addVar(vtype=GRB.BINARY, name=f"r_{idx}")
        self.relax_vars.append(r_var)
        self.obj += weight * r_var
        self.master.setObjective(self.obj, GRB.MINIMIZE)
        self.master.update()

        self._soft.append(list(clause))
        self._weights.append(weight)

        aux = self.next_aux
        self.next_aux += 1
        augmented = list(clause) + [aux]
        self.sat_solver.add_clause(augmented)
        assumption = -aux
        self.assumption_lits.append(assumption)
        self.assumption_to_index[assumption] = idx

    # ------------------------------------------------------------------
    def compute(self) -> None:
        iteration = 0
        while True:
            iteration += 1
            if iteration > self.MAX_ITERATIONS:
                print("Exceeded iteration limit in SAT+MIP hitting-set solver.")
                self.solution = None
                self.cost = float("inf")
                return
            try:
                self.master.optimize()
            except gp.GurobiError as exc:
                print(f"Gurobi error during hitting-set optimization: {exc}")
                self.solution = None
                self.cost = float("inf")
                return

            if self.master.status != GRB.OPTIMAL:
                print("Master MILP failed to reach optimality.")
                self.solution = None
                self.cost = float("inf")
                return

            r_values = [var.X for var in self.relax_vars]
            assumptions = [
                self.assumption_lits[i]
                for i, value in enumerate(r_values)
                if value < 0.5
            ]

            sat = self.sat_solver.solve(assumptions=assumptions)
            if sat:
                model = self.sat_solver.get_model()
                self._extract_solution(model)
                self.cost = self.master.ObjVal
                return

            core = self.sat_solver.get_core()
            if not core:
                print("SAT solver returned empty core; aborting hybrid algorithm.")
                self.solution = None
                self.cost = float("inf")
                return

            try:
                core_indices = sorted({self.assumption_to_index[lit] for lit in core})
            except KeyError:
                print("Encountered assumption literal outside of mapping.")
                self.solution = None
                self.cost = float("inf")
                return

            core_key = tuple(core_indices)
            if core_key in self.seen_cores:
                print("Repeated core detected in hitting-set hybrid; stopping.")
                self.solution = None
                self.cost = float("inf")
                return
            self.seen_cores.add(core_key)

            cut = gp.quicksum(self.relax_vars[i] for i in core_indices)
            self.master.addConstr(cut >= 1, name=f"hit_{iteration}")
            self.master.update()

    # ------------------------------------------------------------------
    def _extract_solution(self, sat_model: List[int] | None) -> None:
        assignment: Dict[int, bool] = {}
        if sat_model is not None:
            for lit in sat_model:
                assignment[abs(lit)] = lit > 0
        self.solution = [var if assignment.get(var, False) else -var for var in range(1, self._nv + 1)]

    # ------------------------------------------------------------------
    @property
    def model(self) -> List[int] | None:
        return self.solution

    # ------------------------------------------------------------------
    def delete(self) -> None:
        if hasattr(self, "master") and self.master is not None:
            self.master.dispose()
            self.master = None
        if hasattr(self, "sat_solver") and self.sat_solver is not None:
            self.sat_solver.delete()
            self.sat_solver = None