from __future__ import annotations

# --- FIXED: Added missing imports here ---
from typing import Dict, List, Sequence
# -----------------------------------------

import gurobipy as gp
from gurobipy import GRB
from pysat.formula import WCNF

class MILPSolver:
    """
    Optimized MILP solver with pseudo-Boolean strengthened constraints.
    Uses batch variable creation and gp.quicksum for performance.
    """

    def __init__(self, wcnf: WCNF):
        self.gurobi_model = gp.Model("pb_strengthened_maxsat")
        self.gurobi_model.setParam("OutputFlag", 1)

        # 1. Batch Create Literal Variables
        # self.x[i] is the Gurobi variable for literal i
        # Using addVars is significantly faster than a loop
        self.x = self.gurobi_model.addVars(
            range(1, wcnf.nv + 1), 
            vtype=GRB.BINARY, 
            name="x"
        )

        self.cost = None
        self.solution = None
        
        # We need to track relaxation variables for the add_clause method
        # addVars returns a tupledict, which is efficient
        self.r = {} 

        self._build_model_from_wcnf(wcnf)

    def _build_model_from_wcnf(self, wcnf: WCNF) -> None:
        # ---------------------------------------------------------
        # A. Hard Clauses
        # ---------------------------------------------------------
        for i, clause in enumerate(wcnf.hard):
            # Generator expression inside quicksum is O(N) faster than loops
            self.gurobi_model.addConstr(
                gp.quicksum(
                    self.x[abs(l)] if l > 0 else (1 - self.x[abs(l)]) 
                    for l in clause
                ) >= 1, 
                name=f"hard_{i}"
            )

        # ---------------------------------------------------------
        # B. Soft Clauses (PB + Cover)
        # ---------------------------------------------------------
        num_soft = len(wcnf.soft)
        
        # Batch create relaxation variables
        self.r = self.gurobi_model.addVars(
            range(num_soft), 
            vtype=GRB.BINARY, 
            name="r"
        )

        # We construct the objective expression once using quicksum
        self.gurobi_model.setObjective(
            gp.quicksum(wcnf.wght[i] * self.r[i] for i in range(num_soft)), 
            GRB.MINIMIZE
        )

        for i, clause in enumerate(wcnf.soft):
            k = len(clause)
            relax_var = self.r[i]

            # Optimization: Extract clause variables ONCE
            # literal_vars contains the Gurobi objects representing "True" for that literal
            literal_vars = [
                self.x[abs(l)] if l > 0 else (1 - self.x[abs(l)]) 
                for l in clause
            ]

            # 1. Satisfaction Lower Bound (Standard PB)
            # sum(literals) + r >= 1
            self.gurobi_model.addConstr(
                gp.quicksum(literal_vars) + relax_var >= 1, 
                name=f"pb_{i}"
            )

            # 2. Falsity Upper Bound (Cover Inequality)
            # sum(NOT literals) <= (k-1) + k*r
            self.gurobi_model.addConstr(
                gp.quicksum(1 - lit for lit in literal_vars) <= (k - 1) + k * relax_var,
                name=f"cover_{i}"
            )

        self.gurobi_model.update()

    # ------------------------------------------------------------------
    def add_clause(self, clause: Sequence[int], weight: int) -> None:
        """Adds a single dynamic clause (slower than batch, but necessary for CEGAR)."""
        idx = len(self.r)
        
        # Add new relaxation variable
        relax_var = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"r_{idx}")
        self.r[idx] = relax_var 

        # Collect literals
        literal_vars = []
        for l in clause:
            var_id = abs(l)
            # Ensure variable exists (in case dynamic clause introduces new var)
            if var_id not in self.x:
                self.x[var_id] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{var_id}")
            
            literal_vars.append(self.x[var_id] if l > 0 else (1 - self.x[var_id]))

        k = len(clause)

        # Add PB constraint
        self.gurobi_model.addConstr(
            gp.quicksum(literal_vars) + relax_var >= 1, 
            name=f"pb_{idx}"
        )

        # Add Cover constraint
        self.gurobi_model.addConstr(
            gp.quicksum(1 - lit for lit in literal_vars) <= (k - 1) + k * relax_var,
            name=f"cover_{idx}"
        )

        # Update Objective incrementally
        current_obj = self.gurobi_model.getObjective()
        current_obj += weight * relax_var
        self.gurobi_model.setObjective(current_obj, GRB.MINIMIZE)
        
        self.gurobi_model.update()

    # ------------------------------------------------------------------
    def compute(self) -> None:
        try:
            self.gurobi_model.optimize()
            if self.gurobi_model.status == GRB.OPTIMAL:
                self.cost = self.gurobi_model.ObjVal
                self.solution = []
                
                # Efficient extraction
                # We iterate up to the max variable index we have created
                max_var_id = max(self.x.keys())
                for var_id in range(1, max_var_id + 1):
                    if var_id in self.x:
                        var = self.x[var_id]
                        if var.X > 0.5:
                            self.solution.append(var_id)
                        else:
                            self.solution.append(-var_id)
            else:
                self.solution = None
                self.cost = float("inf")
        except gp.GurobiError as exc:
            print(f"Gurobi error: {exc}")
            self.solution = None
            self.cost = float("inf")

    @property
    def model(self) -> List[int] | None:
        return self.solution

    def delete(self) -> None:
        if hasattr(self, "gurobi_model") and self.gurobi_model is not None:
            self.gurobi_model.dispose()