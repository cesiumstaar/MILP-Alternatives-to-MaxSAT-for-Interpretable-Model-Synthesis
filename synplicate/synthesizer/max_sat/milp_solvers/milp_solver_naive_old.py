import gurobipy as gp
from gurobipy import GRB
from pysat.formula import WCNF

class MILPSolver:
    """
    A MILP-based solver for MaxSAT problems that mimics the PySAT RC2 interface,
    using Gurobi as the backend.
    """
    def __init__(self, wcnf: WCNF):
        """
        Initializes the solver and builds the MILP model from a WCNF object.
        
        :param wcnf: A WCNF object from the PySAT library.
        """
        self.gurobi_model = gp.Model("wcnf_to_milp")
        self.gurobi_model.setParam('OutputFlag', 1) # suppress Gurobi output for cleaner execution

        # model = gp.Model("LWE_MILP")
        # self.gurobi_model.Params.OutputFlag = 0           # Suppress solver log output
        # self.gurobi_model.Params.PoolSearchMode = 2       # Search for multiple solutio?ns (diverse)
        # self.gurobi_model.Params.PoolSolutions = batch_size  # Max solutions per attempt
        # self.gurobi_model.Params.Threads = 12   # Use multiple CPU threads

        # Set MILP solver hyperparameters
        # self.gurobi_model.Params.Presolve = 2             # Enable aggressive presolving
        # self.gurobi_model.Params.MIPFocus = 1             # Focus on finding feasible solutions
        # self.gurobi_model.Params.Symmetry = 2             # Use symmetry breaking to speed up solving
        # self.gurobi_model.Params.Cuts = 2                 # Use moderate cutting planes

        self.variables = {}  # maps WCNF var ID to Gurobi var object
        self.slack_vars = {} # maps an index to the slack variable for soft clauses
        self.cost = -1
        self.solution = None # will store the model as a list of integers
        
        self._build_model_from_wcnf(wcnf)

    def _build_model_from_wcnf(self, wcnf: WCNF):
        """
        Parses the WCNF object and translates it into a Gurobi MILP model.
        """
        # create a binary variable for each variable in the WCNF formula.
        for var_id in range(1, wcnf.nv + 1):
            self.variables[var_id] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{var_id}")
        
        # set the objective function to minimize.
        obj_expr = gp.LinExpr()
        # add hard clauses as constraints.
        for i, clause in enumerate(wcnf.hard):
            lin_expr = gp.LinExpr()
            for literal in clause:
                var_id = abs(literal)
                if literal > 0:
                    lin_expr += self.variables[var_id]
                else:
                    lin_expr += (1 - self.variables[var_id])
            self.gurobi_model.addConstr(lin_expr >= 1, name=f"hard_{i}")

        # add soft clauses with their weights.
        for i, (clause, weight) in enumerate(zip(wcnf.soft, wcnf.wght)):
            # create a slack variable for each soft clause to indicate if it's unsatisfied.
            # print(i, clause, weight)
            y_i = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"y_{i}")
            self.slack_vars[i] = y_i

            lin_expr = gp.LinExpr()
            for literal in clause:
                var_id = abs(literal)
                if literal > 0:
                    lin_expr += self.variables[var_id]
                else:
                    lin_expr += (1 - self.variables[var_id])
            
            # constraint: (sum of literals) + slack_var >= 1
            self.gurobi_model.addConstr(lin_expr + y_i >= 1, name=f"soft_{i}")
            
            # add the weighted slack variable to the objective.
            obj_expr += weight * y_i
        
        self.gurobi_model.setObjective(obj_expr, GRB.MINIMIZE)
        self.gurobi_model.update()

    def add_clause(self, clause, weight):
        """
        Adds a new weighted clause to the existing Gurobi model.
        This is necessary for handling the lower/upper bound constraints in the `synthesizeWithMILP` function.
        """
        lin_expr = gp.LinExpr()
        for literal in clause:
            var_id = abs(literal)
            # ensure the variable exists in the model
            if var_id not in self.variables:
                self.variables[var_id] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{var_id}")
            
            if literal > 0:
                lin_expr += self.variables[var_id]
            else:
                lin_expr += (1 - self.variables[var_id])
        
        # Create a new slack variable for this dynamic clause.
        num_soft_clauses = len(self.slack_vars)
        y_new = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"y_{num_soft_clauses}")
        self.slack_vars[num_soft_clauses] = y_new

        # add the constraint.
        self.gurobi_model.addConstr(lin_expr + y_new >= 1, name=f"dynamic_clause_{num_soft_clauses}")
        
        # add the weighted slack variable to the objective.
        current_obj = self.gurobi_model.getObjective()
        self.gurobi_model.setObjective(current_obj + weight * y_new, GRB.MINIMIZE)
        self.gurobi_model.update()

    def compute(self):
        """
        Solves the MILP model, calculates the cost, and extracts the model.
        """

        try:
            self.gurobi_model.optimize()
            # requested_threads = self.gurobi_model.getParam("Threads")
            # print(f"Gurobi Threads parameter (requested): {requested_threads}")
            if self.gurobi_model.status == GRB.OPTIMAL:
                self.cost = self.gurobi_model.ObjVal
                # populate the solution list in the same format as RC2.
                self.solution = []
                for var_id in range(1, len(self.variables) + 1):
                    gurobi_var = self.variables[var_id]
                    if gurobi_var.X > 0.5:
                        self.solution.append(var_id)
                    else:
                        self.solution.append(-var_id)
            else:
                # if the problem is infeasible or another issue occurs, set solution to None.
                print("Gurobi solver status is not optimal.")
                self.solution = None
                self.cost = float('inf')
        except gp.GurobiError as e:
            print(f"Gurobi error: {e}")
            self.solution = None
            self.cost = float('inf')
    
    @property
    def model(self):
        """
        Returns the satisfying assignment (model) found by the solver.
        This is a property to mimic the `rc2.model` attribute.
        """
        return self.solution

    def delete(self):
        """
        Disposes of the Gurobi model to free up resources.
        """
        if self.gurobi_model:
            self.gurobi_model.dispose()