from __future__ import annotations

from typing import Dict, List, Sequence, Union

import gurobipy as gp
from gurobipy import GRB
from pysat.formula import WCNF


class MILPSolver:
    """
    Optimized Quadratic-style MILP solver using McCormick linearization.
    Removes redundant 'unsat' indicator variables and uses efficient expression building.
    """

    def __init__(self, wcnf: WCNF):
        self.gurobi_model = gp.Model("quadratic_mccormick_optimized")
        self.gurobi_model.setParam("OutputFlag", 1)

        # 1. Batch create all literal variables (x)
        self.x = self.gurobi_model.addVars(
            range(1, wcnf.nv + 1),
            vtype=GRB.BINARY,
            name="x"
        )

        self.cost: float | None = None
        self.solution: List[int] | None = None
        self.total_weight = sum(wcnf.wght)

        self._build_model_from_wcnf(wcnf)

    def _build_model_from_wcnf(self, wcnf: WCNF) -> None:
        # -----------------------------------------------------------
        # A. Hard Clauses
        # -----------------------------------------------------------
        # Hard clauses must be satisfied (unsat product = 0).
        # It is usually more efficient to just use standard Sum >= 1 for hard clauses
        # unless you specifically want to test the quadratic encoding on them.
        # Here we use the standard sum for performance on hard constraints.
        for i, clause in enumerate(wcnf.hard):
            self.gurobi_model.addConstr(
                gp.quicksum(
                    self.x[abs(l)] if l > 0 else (1 - self.x[abs(l)])
                    for l in clause
                ) >= 1,
                name=f"hard_{i}"
            )

        # -----------------------------------------------------------
        # B. Soft Clauses (Quadratic / McCormick)
        # -----------------------------------------------------------
        num_soft = len(wcnf.soft)

        # Batch create satisfaction indicators (y)
        self.y = self.gurobi_model.addVars(
            range(num_soft),
            vtype=GRB.BINARY,
            name="y"
        )

        # Set Objective immediately (Maximize weighted satisfaction)
        self.gurobi_model.setObjective(
            gp.quicksum(wcnf.wght[i] * self.y[i] for i in range(num_soft)),
            GRB.MAXIMIZE
        )

        # Process clauses
        for i, clause in enumerate(wcnf.soft):
            self._encode_mccormick_chain(i, clause, self.y[i])

        self.gurobi_model.update()

    def _encode_mccormick_chain(self, idx: int, clause: Sequence[int], y_var: gp.Var) -> None:
        """
        Manually builds the McCormick linearization chain for:
        y_var = 1 - Product(unsat_literals)
        """
        # 1. Identify the 'factors' of the product (the falsity expressions)
        # We DO NOT create new variables here. We use the expressions directly.
        # If literal is +x, it is unsat if x=0 (expr: 1-x)
        # If literal is -x, it is unsat if x=1 (expr: x)
        factors = []
        for l in clause:
            var = self.x[abs(l)]
            factors.append((1 - var) if l > 0 else var)

        if len(factors) == 1:
            # Direct link: y = 1 - unsat => y = satisfy
            # If factor is (1-x), y = 1 - (1-x) = x
            self.gurobi_model.addConstr(y_var == (1 - factors[0]), name=f"link_{idx}")
            return

        # 2. Build intermediate variables for the product chain
        # For factors [A, B, C], we need P1 = A*B, P2 = P1*C.
        # We need len(clause) - 1 intermediate variables.
        # We call them 'p' (product accumulation)
        num_intermediates = len(factors) - 1
        p_vars = self.gurobi_model.addVars(
            range(num_intermediates),
            vtype=GRB.BINARY,
            name=f"p_{idx}"
        ).values() # extract list from tupledict

        # 3. McCormick Linearization Loop
        # Current 'product' starts as the first factor
        current_prod = factors[0]

        for step, (next_factor, p_var) in enumerate(zip(factors[1:], p_vars)):
            # We want: p_var = current_prod * next_factor
            # Constraints:
            # 1. p <= current_prod
            # 2. p <= next_factor
            # 3. p >= current_prod + next_factor - 1

            self.gurobi_model.addConstr(p_var <= current_prod, name=f"mc_{idx}_{step}_1")
            self.gurobi_model.addConstr(p_var <= next_factor,  name=f"mc_{idx}_{step}_2")
            self.gurobi_model.addConstr(p_var >= current_prod + next_factor - 1, name=f"mc_{idx}_{step}_3")

            # Update for next iteration
            current_prod = p_var

        # 4. Final Link
        # The last p_var represents the product of ALL unsat indicators.
        # y_var (satisfaction) = 1 - product
        self.gurobi_model.addConstr(y_var == 1 - current_prod, name=f"final_link_{idx}")

    def compute(self) -> None:
        try:
            self.gurobi_model.optimize()
            if self.gurobi_model.status == GRB.OPTIMAL:
                satisfied_weight = self.gurobi_model.ObjVal
                self.cost = self.total_weight - satisfied_weight
                self.solution = []
                max_var = max(self.x.keys())
                for i in range(1, max_var + 1):
                    if i in self.x:
                        self.solution.append(i if self.x[i].X > 0.5 else -i)
            else:
                self.solution = None
                self.cost = float("inf")
        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")
            self.cost = float("inf")

    @property
    def model(self) -> List[int] | None:
        return self.solution

    def delete(self) -> None:
        if self.gurobi_model:
            self.gurobi_model.dispose()