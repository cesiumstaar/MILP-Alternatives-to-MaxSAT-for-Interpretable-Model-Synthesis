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
        self.gurobi_model.setParam('OutputFlag', 1)  # 0 to suppress solver output
        # self.gurobi_model.Params.Threads = 12        # Multi-threaded optimization

        self.variables = {}   # Maps WCNF var ID to Gurobi var object
        self.clause_vars = {} # Maps clause index to its satisfaction variable y_i
        self.cost = -1
        self.solution = None  # Will store the model as a list of integers

        self._build_model_from_wcnf(wcnf)

    # ----------------------------------------------------------------------
    def _build_model_from_wcnf(self, wcnf: WCNF):
        """
        Parses the WCNF object and translates it into a Gurobi MILP model using
        the clause satisfaction encoding.
        """
        # 1. Create binary variable for each SAT variable
        for var_id in range(1, wcnf.nv + 1):
            self.variables[var_id] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{var_id}")

        obj_expr = gp.LinExpr()

        # 2. Encode hard clauses: must be satisfied
        for i, clause in enumerate(wcnf.hard):
            lin_expr = gp.LinExpr()
            for literal in clause:
                var_id = abs(literal)
                if literal > 0:
                    lin_expr += self.variables[var_id]
                else:
                    lin_expr += (1 - self.variables[var_id])
            self.gurobi_model.addConstr(lin_expr >= 1, name=f"hard_{i}")

        # 3. Encode soft clauses: introduce y_i for satisfaction indicator
        for i, (clause, weight) in enumerate(zip(wcnf.soft, wcnf.wght)):
            y_i = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"y_{i}")
            self.clause_vars[i] = y_i

            sum_expr = gp.LinExpr()
            for literal in clause:
                var_id = abs(literal)
                if var_id not in self.variables:
                    self.variables[var_id] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{var_id}")
                x = self.variables[var_id]

                # Each literal constrains y_i
                if literal > 0:
                    self.gurobi_model.addConstr(x <= y_i, name=f"soft_{i}_lit_{var_id}_pos")
                    sum_expr += x
                else:
                    self.gurobi_model.addConstr(1 - x <= y_i, name=f"soft_{i}_lit_{var_id}_neg")
                    sum_expr += (1 - x)

            # Ensure y_i cannot be 1 unless at least one literal is true
            self.gurobi_model.addConstr(sum_expr >= y_i, name=f"soft_{i}_sum")

            # Add to objective (maximize satisfied clause weight)
            obj_expr += weight * y_i

        # 4. Set the objective to maximize total satisfied weight
        self.gurobi_model.setObjective(obj_expr, GRB.MAXIMIZE)
        self.gurobi_model.update()

    # ----------------------------------------------------------------------
    def add_clause(self, clause, weight):
        """
        Dynamically add a new weighted clause (soft) using the same encoding.
        """
        y_new = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"y_{len(self.clause_vars)}")
        self.clause_vars[len(self.clause_vars)] = y_new

        sum_expr = gp.LinExpr()
        for literal in clause:
            var_id = abs(literal)
            if var_id not in self.variables:
                self.variables[var_id] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{var_id}")
            x = self.variables[var_id]

            if literal > 0:
                self.gurobi_model.addConstr(x <= y_new)
                sum_expr += x
            else:
                self.gurobi_model.addConstr(1 - x <= y_new)
                sum_expr += (1 - x)

        self.gurobi_model.addConstr(sum_expr >= y_new)

        # Update objective: maximize satisfied weight
        current_obj = self.gurobi_model.getObjective()
        self.gurobi_model.setObjective(current_obj + weight * y_new, GRB.MAXIMIZE)
        self.gurobi_model.update()

    # ----------------------------------------------------------------------
    def compute(self):
        """
        Solves the MILP model, calculates the total satisfied weight, and extracts the model.
        """
        try:
            self.gurobi_model.optimize()
            if self.gurobi_model.status == GRB.OPTIMAL:
                self.cost = self.gurobi_model.ObjVal
                self.solution = []
                for var_id in range(1, len(self.variables) + 1):
                    gurobi_var = self.variables[var_id]
                    if gurobi_var.X > 0.5:
                        self.solution.append(var_id)
                    else:
                        self.solution.append(-var_id)
            else:
                print("Gurobi solver status is not optimal.")
                self.solution = None
                self.cost = float('-inf')
        except gp.GurobiError as e:
            print(f"Gurobi error: {e}")
            self.solution = None
            self.cost = float('-inf')

    # ----------------------------------------------------------------------
    @property
    def model(self):
        """Returns the satisfying assignment (model) found by the solver."""
        return self.solution

    # ----------------------------------------------------------------------
    def delete(self):
        """Disposes of the Gurobi model to free up resources."""
        if self.gurobi_model:
            self.gurobi_model.dispose()
