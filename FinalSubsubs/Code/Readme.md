
# Synplicated

This repository bundles the assets needed to reproduce the Synplicate experiments. The steps below summarize how to build the required toolchain, prepare experiment inputs, run Synplicate, and read the results.

## Build the Synplicate toolchain

Synplicate itself is maintained at [torfah/synplicate](https://github.com/torfah/synplicate); follow the standard build steps from that repository (Python 3 environment plus the bundled `bc2cnf` MaxSAT encoder). In addition to those upstream instructions, apply the project-specific steps here:

1. Set up Synplicate as described in the upstream README, including compiling `bc2cnf` and installing the Python dependencies (`pandas`, `python-sat`, `tabulate`, and any solver-specific extras such as `gurobipy` when needed). (or you can use the configure.sh script to do it all at once and create a venv)

2. Copy `erm_procedure.py` into `synplicate/methods/erm/`.

3. Copy the provided `max_sat` folder into `synplicate/synthesizer/` (replacing the upstream directory if present).

4. Copy `synplicate.py` and `simple_solver` into the top-level `synplicate/` directory.

5. Download the experiments bundle from <https://github.com/hotramen-hellfire/Experiments-for-synplicate>, place it inside `synplicate/`, and rename the folder to `experiments` so it contains subfolders such as `theorem_prover/` and `load_acquisition/`.

6. create a symbolic link to `experiments/` in `simple_solver`, you can run `run_experiments.py` for the experiment/ observation mentioned in the report.

After these steps, you can use the existing `synplicate.py` entry point or the convenience runner described below. No additional Makefile is required beyond the upstream build process.

## Prepare inputs

Experiment inputs are organized under `synplicate/experiments/`. Each experiment folder contains its feature definitions, sampler, configuration, and any pretrained models. For a minimal check:

```bash

cd synplicate

python synplicate.py experiments/Iris/

```

This runs the Iris example using the predicates and sampler bundled in the downloaded experiments package.

## Run experiments efficiently

For batch execution and standardized logging, use `run_experiments.py` from the repository root:

```bash

cd synplicate

python run_experiments.py

```

The script iterates through the experiments listed in `EXPERIMENTS_TO_RUN`, executes the configured solvers, and stores console output in `term_output/`. You can adjust which experiments or solvers run by editing the `EXPERIMENTS_TO_RUN` and `SOLVER_CONFIG` dictionaries inside `run_experiments.py`.

## Interpret the results

During a run, Synplicate prints metrics such as synthesis time, accuracy, explainability, and total cost. The batch runner also parses these fields and records per-experiment summaries in `experiment_results.json`. A typical entry looks like:

```json

{

"Iris": {

"default": {"time": 12.3, "acc": 0.94, "exp": 0.81, "cost": 42.0, "corR": 210, "expR": 180}

}

}

```

Higher accuracy and explainability values indicate better model faithfulness and interpretability, while lower total cost reflects improved trade-offs in the weighted MaxSAT formulation. Use these metrics to compare solvers across experiments; the `term_output/` logs provide the raw command output if you need to inspect any specific run.