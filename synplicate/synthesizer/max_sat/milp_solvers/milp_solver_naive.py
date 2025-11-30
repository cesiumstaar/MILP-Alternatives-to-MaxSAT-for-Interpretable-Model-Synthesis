import gurobipy as gp
from gurobipy import GRB
from pysat.formula import WCNF

class MILPSolver:
    """
    Optimized MILP solver using gp.quicksum and batch variable creation.
    """
    def __init__(self, wcnf: WCNF):
        self.gurobi_model = gp.Model("optimized_maxsat")
        self.gurobi_model.setParam('OutputFlag', 1)

        # Optimization: Create all literal variables in one batch (C-level call)
        # This returns a tupledict which is faster than a standard dict for Gurobi
        self.variables = self.gurobi_model.addVars(
            range(1, wcnf.nv + 1), 
            vtype=GRB.BINARY, 
            name="x"
        )
        
        self.slack_vars = {} 
        self.cost = -1
        self.solution = None
        
        self._build_model_from_wcnf(wcnf)

    def _build_model_from_wcnf(self, wcnf: WCNF):
        # 1. Add Hard Clauses (No slack variables)
        # We use a generator expression inside quicksum for speed
        for i, clause in enumerate(wcnf.hard):
            self.gurobi_model.addConstr(
                gp.quicksum(
                    self.variables[abs(l)] if l > 0 else (1 - self.variables[abs(l)]) 
                    for l in clause
                ) >= 1,
                name=f"hard_{i}"
            )

        # 2. Add Soft Clauses (With slack variables)
        # Create all slack variables in one batch
        num_soft = len(wcnf.soft)
        slack_indices = list(range(num_soft))
        
        # addVars is much faster than calling addVar in a loop
        self.slack_vars_tupledict = self.gurobi_model.addVars(
            slack_indices, 
            vtype=GRB.BINARY, 
            name="slack"
        )
        
        # We need to map these back to your dictionary format if you rely on self.slack_vars[i]
        # (Though using the tupledict directly is faster)
        self.slack_vars = self.slack_vars_tupledict

        # 3. Build Soft Constraints
        for i, (clause, weight) in enumerate(zip(wcnf.soft, wcnf.wght)):
            slack_var = self.slack_vars[i]
            
            # Use quicksum to build the clause expression
            # (sum of literals) + slack >= 1
            self.gurobi_model.addConstr(
                gp.quicksum(
                    self.variables[abs(l)] if l > 0 else (1 - self.variables[abs(l)])
                    for l in clause
                ) + slack_var >= 1,
                name=f"soft_{i}"
            )

        # 4. Set Objective (Minimize weighted sum of slacks)
        # using quicksum here is the biggest performance booster
        self.gurobi_model.setObjective(
            gp.quicksum(
                wcnf.wght[i] * self.slack_vars[i] 
                for i in range(num_soft)
            ),
            GRB.MINIMIZE
        )
        
        self.gurobi_model.update()

    def add_clause(self, clause, weight):
        """
        Adds a single dynamic clause.
        NOTE: Dynamic addition is inherently slower than batch addition.
        """
        # Ensure variables exist (only check if necessary)
        for literal in clause:
            var_id = abs(literal)
            if var_id not in self.variables:
                self.variables[var_id] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{var_id}")
        
        # Create new slack variable
        idx = len(self.slack_vars)
        new_slack = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"slack_{idx}")
        self.slack_vars[idx] = new_slack

        # Build constraint using quicksum
        self.gurobi_model.addConstr(
            gp.quicksum(
                self.variables[abs(l)] if l > 0 else (1 - self.variables[abs(l)])
                for l in clause
            ) + new_slack >= 1,
            name=f"dyn_{idx}"
        )

        # Update objective
        # Getting the expression and adding to it is okay for single updates
        current_obj = self.gurobi_model.getObjective()
        current_obj += weight * new_slack
        self.gurobi_model.setObjective(current_obj, GRB.MINIMIZE)
        
        # Only update if strictly necessary for immediate solving
        self.gurobi_model.update()

    def compute(self):
        try:
            self.gurobi_model.optimize()
            if self.gurobi_model.status == GRB.OPTIMAL:
                self.cost = self.gurobi_model.ObjVal
                self.solution = []
                # extracting solution efficiently
                for var_id, var_obj in self.variables.items():
                    # Gurobi vars are objects, need to check .X
                    if var_obj.X > 0.5:
                        self.solution.append(var_id)
                    else:
                        self.solution.append(-var_id)
            else:
                print("Status not optimal")
                self.solution = None
                self.cost = float('inf')
        except gp.GurobiError as e:
            print(f"Error: {e}")
            self.solution = None
            self.cost = float('inf')

    @property
    def model(self):
        return self.solution

    def delete(self):
        if hasattr(self, 'gurobi_model') and self.gurobi_model:
            self.gurobi_model.dispose()