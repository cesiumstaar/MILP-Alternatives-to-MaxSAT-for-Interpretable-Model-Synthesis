from __future__ import annotations

from typing import Dict, List, Sequence

import gurobipy as gp
from gurobipy import GRB
from pysat.formula import WCNF


class MILPSolver:
    """
    Optimized Extended Formulation Solver.
    Matches original logic (Minimization) exactly but uses vectorization for speed.
    """
    def __init__(self, wcnf: WCNF):
        self.gurobi_model = gp.Model("extended_formulation_maxsat")
        self.gurobi_model.setParam("OutputFlag", 1)
        
        # 1. Batch Create Literal Variables
        # We use a tupledict, but we will extract carefully to ensure order
        self.x = self.gurobi_model.addVars(
            range(1, wcnf.nv + 1), 
            vtype=GRB.BINARY, 
            name="x"
        )

        self.cost = None
        self.solution = None
        
        # Track relaxation variables for dynamic addition
        self.y = {}
        
        self._build_model_from_wcnf(wcnf)

    def _build_model_from_wcnf(self, wcnf: WCNF):
        # ---------------------------------------------------------
        # A. Handle Hard Clauses (Standard Sum >= 1)
        # ---------------------------------------------------------
        # We skip creating 'y' variables for hard clauses to save memory.
        # Enforcing Sum >= 1 is mathematically equivalent to y=1.
        for i, clause in enumerate(wcnf.hard):
            self.gurobi_model.addConstr(
                gp.quicksum(
                    self.x[abs(l)] if l > 0 else (1 - self.x[abs(l)])
                    for l in clause
                ) >= 1,
                name=f"hard_{i}"
            )

        # ---------------------------------------------------------
        # B. Handle Soft Clauses (Extended Formulation)
        # ---------------------------------------------------------
        num_soft = len(wcnf.soft)
        
        # 1. Batch Create Indicator Variables (y)
        # y[i] = 1 if clause i is satisfied
        self.y = self.gurobi_model.addVars(
            range(num_soft), 
            vtype=GRB.BINARY, 
            name="y"
        )

        # 2. Build Objective (Minimize Violation)
        # Original Logic: Minimize sum( weight * (1 - y) )
        # This is robust against floating point issues in 'Total - Sat'
        obj_expr = gp.quicksum(
            wcnf.wght[i] * (1 - self.y[i]) 
            for i in range(num_soft)
        )
        
        self.gurobi_model.setObjective(obj_expr, GRB.MINIMIZE)

        # 3. Add Extended Constraints
        for i, clause in enumerate(wcnf.soft):
            k = len(clause)
            y_var = self.y[i]
            
            # Extract literal variables for this clause
            # (Pre-extracting list is faster than generator in loop)
            lits = [
                self.x[abs(l)] if l > 0 else (1 - self.x[abs(l)])
                for l in clause
            ]
            
            sum_literals = gp.quicksum(lits)
            
            # Constraint 1: Upper Bound (Implication)
            # If Sum=0, y MUST be 0. (y <= Sum)
            self.gurobi_model.addConstr(y_var <= sum_literals, name=f"soft_ub_{i}")

            # Constraint 2: Lower Bound (Tightening)
            # y >= Sum / k
            self.gurobi_model.addConstr(y_var * k >= sum_literals, name=f"soft_lb_{i}")

        self.gurobi_model.update()

    # ------------------------------------------------------------------
    def add_clause(self, clause: Sequence[int], weight: int) -> None:
        """Adds a single dynamic clause."""
        idx = len(self.y)
        
        # Create new indicator variable
        y_var = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"y_{idx}")
        self.y[idx] = y_var

        # Collect literals
        lits = []
        for l in clause:
            var_id = abs(l)
            if var_id not in self.x:
                self.x[var_id] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{var_id}")
            lits.append(self.x[var_id] if l > 0 else (1 - self.x[var_id]))
        
        sum_literals = gp.quicksum(lits)
        k = len(clause)

        # Add Constraints
        self.gurobi_model.addConstr(y_var <= sum_literals, name=f"dyn_ub_{idx}")
        self.gurobi_model.addConstr(y_var * k >= sum_literals, name=f"dyn_lb_{idx}")

        # Update Objective (Minimize Violation)
        current_obj = self.gurobi_model.getObjective()
        current_obj += weight * (1 - y_var)
        self.gurobi_model.setObjective(current_obj, GRB.MINIMIZE)
        
        self.gurobi_model.update()

    # ------------------------------------------------------------------
    def compute(self) -> None:
        try:
            self.gurobi_model.optimize()
            if self.gurobi_model.status == GRB.OPTIMAL:
                # Direct Cost extraction (matches Original logic)
                self.cost = self.gurobi_model.ObjVal
                
                self.solution = []
                # SAFETY FIX: Ensure strict sorting of variable IDs
                # Iterating dict items is mostly ordered, but sorting ensures determinism
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
            self.cost = float("inf")

    # ------------------------------------------------------------------
    @property
    def model(self) -> List[int] | None:
        return self.solution

    # ------------------------------------------------------------------
    def delete(self) -> None:
        if hasattr(self, "gurobi_model") and self.gurobi_model is not None:
            self.gurobi_model.dispose()