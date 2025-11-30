"""
Hierarchical MILP with Logic-Based Benders Decomposition
=======================================================

This solver separates optimization (relaxation selection) from feasibility
checking using a logic-based Benders framework (see Hooker & Ottosson, 2003; Codato
& Fischetti, 2006). The master MILP chooses which soft clauses to relax via
binary variables r_j and minimizes their weighted sum. For a tentative master
solution, a SAT subproblem—implemented with PySAT's Minisat22—checks whether the
selected clauses admit a feasible assignment. If infeasible, the unsatisfied set
of clauses returned as an UNSAT core yields a Benders cut of the form
\sum_{j \in core} r_j >= 1, excluding that relaxation pattern. The process
iterates until a feasible assignment is found, closely mirroring logic-based
Benders algorithms for MaxSAT/Max-CSP.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import gurobipy as gp
from gurobipy import GRB
from pysat.formula import WCNF
from pysat.solvers import Minisat22


class MILPSolver:
    """Logic-based Benders MILP solver with SAT subproblem checks."""

    def __init__(self, wcnf: WCNF):
        self.master = gp.Model("benders_maxsat")
        self.master.setParam("OutputFlag", 1)

        self.relax_vars: List[gp.Var] = []
        self.obj_expr = gp.LinExpr()
        self.cost: float | None = None
        self.solution: List[int] | None = None

        self.original_var_count = wcnf.nv
        self.cnf_clauses: List[List[int]] = [list(cl) for cl in wcnf.hard]
        self.assumption_lits: List[int] = []
        self.assumption_to_index: Dict[int, int] = {}
        self.next_aux_var = self.original_var_count + 1

        for clause, weight in zip(wcnf.soft, wcnf.wght):
            self._add_soft_clause(clause, weight)

        if self.relax_vars:
            self.master.setObjective(self.obj_expr, GRB.MINIMIZE)
        else:
            self.master.setObjective(gp.LinExpr(0.0))
        self.master.update()

    # ------------------------------------------------------------------
    def _add_soft_clause(self, clause: Sequence[int], weight: int) -> None:
        idx = len(self.relax_vars)
        r_var = self.master.addVar(vtype=GRB.BINARY, name=f"r_{idx}")
        self.relax_vars.append(r_var)
        self.obj_expr += weight * r_var

        aux_var = self.next_aux_var
        self.next_aux_var += 1
        clause_with_aux = list(clause) + [aux_var]
        self.cnf_clauses.append(clause_with_aux)

        assumption = -aux_var
        self.assumption_lits.append(assumption)
        self.assumption_to_index[assumption] = idx

    # ------------------------------------------------------------------
    def add_clause(self, clause: Sequence[int], weight: int) -> None:
        self._add_soft_clause(clause, weight)
        self.master.setObjective(self.obj_expr, GRB.MINIMIZE)
        self.master.update()

    # ------------------------------------------------------------------
    def compute(self) -> None:
        iteration = 0
        while True:
            iteration += 1
            try:
                self.master.optimize()
            except gp.GurobiError as exc:
                print(f"Gurobi error during master optimization: {exc}")
                self.solution = None
                self.cost = float("inf")
                return

            if self.master.status != GRB.OPTIMAL:
                print("Master MILP did not converge to optimality.")
                self.solution = None
                self.cost = float("inf")
                return

            r_values = [var.X for var in self.relax_vars]
            assumptions = [
                self.assumption_lits[i]
                for i, value in enumerate(r_values)
                if value < 0.5
            ]

            with Minisat22(bootstrap_with=self.cnf_clauses) as solver:
                sat = solver.solve(assumptions=assumptions)
                if sat:
                    model = solver.get_model()
                    self._extract_solution(model)
                    self.cost = self.master.ObjVal
                    return

                core = solver.get_core()

            if not core:
                print("SAT subproblem returned empty core; aborting.")
                self.solution = None
                self.cost = float("inf")
                return

            try:
                core_indices = sorted({self.assumption_to_index[lit] for lit in core})
            except KeyError:
                print("Encountered assumption literal outside mapping; aborting.")
                self.solution = None
                self.cost = float("inf")
                return

            cut_expr = gp.quicksum(self.relax_vars[i] for i in core_indices)
            self.master.addConstr(cut_expr >= 1, name=f"benders_cut_{iteration}")
            self.master.update()

    # ------------------------------------------------------------------
    def _extract_solution(self, sat_model: List[int] | None) -> None:
        assignment = {}
        if sat_model is not None:
            for lit in sat_model:
                assignment[abs(lit)] = lit > 0
        self.solution = []
        for var in range(1, self.original_var_count + 1):
            val = assignment.get(var, False)
            self.solution.append(var if val else -var)

    # ------------------------------------------------------------------
    @property
    def model(self) -> List[int] | None:
        return self.solution

    # ------------------------------------------------------------------
    def delete(self) -> None:
        if hasattr(self, "master") and self.master is not None:
            self.master.dispose()