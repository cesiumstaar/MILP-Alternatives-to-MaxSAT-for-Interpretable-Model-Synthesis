import os
import subprocess
import re
import json
from tabulate import tabulate
import time
import shutil

# --- Configuration ---

EXPERIMENTS_TO_RUN = {
    "theorem_prover": 1,
    "california_census": 1,
    "Iris3": 1,
    "Banknote3": 1,
    "ICML/AutoTaxi": 1,
    "Adult3_1": 1,
    "loan_acquisition": 1,
    "Adult3": 1,
    "Banknote4": 1,
    "Iris": 1,
}

# 'interleaved' or 'grouped'
RUNNING_STRATEGY = 'interleaved'
BASE_PATH = "experiments/"
CHECKPOINT_FILE = "experiment_results.json"
OUTPUT_LOG_DIR = "term_output"
ERROR_LOG_DIR = os.path.join(OUTPUT_LOG_DIR, "errors")
SNAPSHOT_DIR_NAME = "previous_runs"

# Timeout (seconds). 0 or None => no timeout.
TIMEOUT_SECONDS = 600  # e.g., 600 for 10 minutes, 0 for unlimited.

# --- View mode ---
# "complete" -> full metrics in table
# "compact"  -> only time per solver in table
# "linear"   -> no table, linear listing per experiment
VIEW = "linear"  # change to "complete" or "compact" or "linear" as needed

# --- Per-solver toggle (similar to EXPERIMENTS_TO_RUN) ---
# 1 = run this solver, 0 = skip it
SOLVER_CONFIG = {
    "default": 1,                         # RC2 / default MaxSAT solver
    # MILP-style solvers (passed as names like "milp_solver_naive")
    # "milp_solver_naive_old": 0,
    "milp_solver_naive": 1,
    # NEW solvers (passed as e.g. --solver="solver_csp.py")
    # "solver_big_m_lp": 0,
    # "solver_csp": 0,
    # "solver_max_cut": 0,
    # "solver_pseudo_boolean": 0,
    # "solver_qubo": 0,
    # "solver_sat_mip_hitting_set": 0,

    # "milp_solver_big_m": 0,
    "milp_solver_extended_formulation": 1,
    # "milp_solver_naive2": 0,
    "milp_solver_pb_strengthened": 1,
    "milp_solver_quadratic": 1,

    # "milp_solver_benders": 0,
    # "milp_solver_core_guided": 0,
}

# This will be populated in main() from SOLVER_CONFIG
SOLVERS = []


# --- Helper Functions ---

def parse_output(output_string):
    """Parses the stdout of the script to find the required metrics."""
    try:
        time_match = re.search(r"Synthesis time:\s*([\d.]+)", output_string)
        acc_match = re.search(r"Accuracy:\s*([\d.]+)", output_string)
        exp_match = re.search(r"Explainability:\s*([\d.]+)", output_string)
        cost_match = re.search(r"Total cost:\s*([\d.]+)", output_string)

        cor_reward_match = re.search(r"correctnessReward:\s*([\d]+)", output_string)
        exp_reward_match = re.search(r"explainibilityReward:\s*([\d]+)", output_string)

        if (time_match and acc_match and exp_match and cost_match and
                cor_reward_match and exp_reward_match):

            return {
                "time": float(time_match.group(1)),
                "acc": float(acc_match.group(1)),
                "exp": float(exp_match.group(1)),
                "cost": float(cost_match.group(1)),
                "corR": int(cor_reward_match.group(1)),
                "expR": int(exp_reward_match.group(1)),
            }

    except (TypeError, ValueError):
        return None
    return None


def display_table(results, timer_str=None):
    """
    Table-based view (used for VIEW == 'complete' or 'compact').
    """
    os.system('cls' if os.name == 'nt' else 'clear')

    headers = ["Experiment"]
    for solver in SOLVERS:
        if VIEW == "complete":
            headers.append(f"{solver}\n<Time, Cost, [Acc, Exp], corR, expR>")
        else:  # compact
            headers.append(f"{solver}\n<Time>")

    table_data = []

    for exp_name in EXPERIMENTS_TO_RUN:
        row = [exp_name]
        for solver in SOLVERS:
            res = results.get(exp_name, {}).get(solver, "Not Run")
            if isinstance(res, dict):
                if VIEW == "complete":
                    formatted_res = (
                        f"<{res['time']:.2f}s, {int(res['cost'])}, "
                        f"[{res['acc']:.2f}, {res['exp']:.2f}], "
                        f"corR: {res['corR']}, expR: {res['expR']}>"
                    )
                else:  # compact -> only time
                    formatted_res = f"{res['time']:.2f}s"
                row.append(formatted_res)
            else:
                row.append(res)
        table_data.append(row)

    print("--- Synplicate Experiment Runner ---")
    print(tabulate(table_data, headers=headers, stralign="center", maxcolwidths=40))
    print("\nPress Ctrl+C while a run is active to SKIP that experiment;")
    print("press Ctrl+C between runs to pause/stop the whole script.")
    if timer_str:
        print(timer_str)
    print("-" * 30)


def display_linear(results, current_exp=None, current_solver=None, elapsed=None):
    """
    Linear (non-table) view for VIEW == 'linear'.
    Shows all experiments and solvers line by line.
    """
    os.system('cls' if os.name == 'nt' else 'clear')

    print("--- Synplicate Experiment Runner (Linear View) ---\n")

    for exp_name in EXPERIMENTS_TO_RUN:
        print(f"Experiment: {exp_name}")
        for solver in SOLVERS:
            res = results.get(exp_name, {}).get(solver, "Not Run")
            prefix = f"  - {solver}: "
            if isinstance(res, dict):
                # Show full metrics in one line (compact but informative)
                line = (
                    f"time={res['time']:.2f}s, cost={int(res['cost'])} "
                    # f"acc={res['acc']:.2f}, exp={res['exp']:.2f}, "
                    # f"corR={res['corR']}, expR={res['expR']}"
                )
            else:
                line = str(res)
            print(prefix + line)
        print()

    if current_exp is not None and current_solver is not None and elapsed is not None:
        print(f"Currently running: {current_exp} [{current_solver}] | Elapsed: {int(elapsed)}s")

    print("\nPress Ctrl+C while a run is active to SKIP that experiment;")
    print("press Ctrl+C between runs to pause/stop the whole script.")
    print("-" * 30)


def save_results(results):
    """Saves the current results dictionary to the checkpoint file."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(results, f, indent=4)


def load_results():
    """Loads results from the checkpoint file if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def show_results(results, timer_str=None, current_exp=None, current_solver=None, elapsed=None):
    """
    Wrapper to show results according to VIEW.
    """
    if VIEW == "linear":
        display_linear(results, current_exp=current_exp, current_solver=current_solver, elapsed=elapsed)
    else:
        display_table(results, timer_str=timer_str)


def copy_error_log(log_filename):
    """
    Copy the log file to the errors subfolder for easier inspection.
    """
    try:
        os.makedirs(ERROR_LOG_DIR, exist_ok=True)
        shutil.copy(log_filename, os.path.join(ERROR_LOG_DIR, os.path.basename(log_filename)))
    except Exception:
        # Don't crash on copy errors; logging is best-effort.
        pass


def run_single_experiment(exp_name, solver, results):
    """Handles the logic for running one experiment and updating results."""
    # If this experiment+solver already has a dict of results, skip rerun
    if isinstance(results.get(exp_name, {}).get(solver), dict):
        return results

    results[exp_name][solver] = "Running..."
    show_results(results, current_exp=exp_name, current_solver=solver, elapsed=0)

    path = os.path.join(BASE_PATH, exp_name)
    command = ["python3", "synplicate.py", f"{path}/", f"--solver={solver}"]

    start_time = time.time()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    timed_out = False
    skipped = False
    elapsed = 0

    # Poll process, update timer, and enforce timeout if set
    try:
        while True:
            if process.poll() is not None:
                # Process finished
                break

            elapsed = time.time() - start_time

            if TIMEOUT_SECONDS and elapsed > TIMEOUT_SECONDS:
                timed_out = True
                process.kill()
                break

            timer_display = f"Elapsed: {int(elapsed)}s"
            if VIEW == "linear":
                show_results(results, current_exp=exp_name, current_solver=solver, elapsed=elapsed)
                print(f"Executing: {' '.join(command)}")
            else:
                show_results(results, timer_str=timer_display)
                print(f"Executing: {' '.join(command)}")
            time.sleep(1)

    except KeyboardInterrupt:
        # Treat this as a "skip current experiment" command
        skipped = True
        try:
            process.kill()
        except Exception:
            pass

    stdout, stderr = process.communicate()

    safe_exp_name = exp_name.replace('/', '_')
    log_filename = os.path.join(OUTPUT_LOG_DIR, f"{safe_exp_name}_{solver}.log")
    with open(log_filename, 'w') as log_file:
        log_file.write(f"--- Command ---\n{' '.join(command)}\n\n")
        if timed_out:
            log_file.write(f"--- Status: TIMEOUT after {int(elapsed)}s ---\n\n")
        elif skipped:
            log_file.write(f"--- Status: SKIPPED by user after {int(elapsed)}s ---\n\n")
        else:
            log_file.write(f"--- Return Code: {process.returncode} ---\n\n")
        log_file.write("--- Standard Output (stdout) ---\n")
        log_file.write(stdout)
        log_file.write("\n\n--- Standard Error (stderr) ---\n")
        log_file.write(stderr)

    # Handle skip / timeout paths first
    if skipped:
        results[exp_name][solver] = "Skipped by user"
        save_results(results)
        copy_error_log(log_filename)  # treat skip as "special" log
        show_results(results)
        return results

    if timed_out:
        results[exp_name][solver] = f"Timeout after {int(elapsed)}s"
        save_results(results)
        copy_error_log(log_filename)
        show_results(results)
        return results

    # Normal completion path
    returncode = process.returncode

    if returncode == 0:
        parsed_data = parse_output(stdout)
        if parsed_data:
            results[exp_name][solver] = parsed_data
        else:
            results[exp_name][solver] = "Parse Error"
            copy_error_log(log_filename)
    else:
        results[exp_name][solver] = f"Run Error (Code: {returncode})"
        copy_error_log(log_filename)

    save_results(results)
    # Snapshot the synthesized program folder (if it exists) for inspection.
    try:
        exp_dir = os.path.join(BASE_PATH, exp_name)
        program_dir = os.path.join(exp_dir, "program")
        if os.path.isdir(program_dir):
            snapshot_root = os.path.join(exp_dir, SNAPSHOT_DIR_NAME)
            os.makedirs(snapshot_root, exist_ok=True)
            timestamp = int(time.time())
            snapshot_dir = os.path.join(snapshot_root, f"program_{solver}_{timestamp}")
            shutil.copytree(program_dir, snapshot_dir)
    except Exception:
        # Best-effort; don't disrupt experiment flow if snapshot fails.
        pass

    # After completion, flush the updated results
    show_results(results)
    return results


# --- Main Execution ---

def main():
    global SOLVERS

    os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)
    os.makedirs(ERROR_LOG_DIR, exist_ok=True)
    results = load_results()

    # Determine active solvers from SOLVER_CONFIG
    SOLVERS = [name for name, run in SOLVER_CONFIG.items() if run == 1]

    if results:
        show_results(results)
        try:
            choice = input("Previous results found. [C]ontinue or [S]tart fresh? ").strip().lower()
        except EOFError:
            # If user sends EOF (Ctrl+D), just continue with existing results
            choice = 'c'
        if choice.startswith('s'):
            print("Starting fresh, clearing previous results...")
            results = {}
            time.sleep(1)

    # Ensure each experiment has an entry
    for exp_name in EXPERIMENTS_TO_RUN:
        if exp_name not in results:
            results[exp_name] = {}

    try:
        active_experiments = [name for name, run in EXPERIMENTS_TO_RUN.items() if run == 1]

        if RUNNING_STRATEGY == 'interleaved':
            for exp_name in active_experiments:
                for solver in SOLVERS:
                    results = run_single_experiment(exp_name, solver, results)

        elif RUNNING_STRATEGY == 'grouped':
            for solver in SOLVERS:
                for exp_name in active_experiments:
                    results = run_single_experiment(exp_name, solver, results)
        else:
            print(f"Error: Unknown RUNNING_STRATEGY '{RUNNING_STRATEGY}'. Please use 'interleaved' or 'grouped'.")
            return

    except KeyboardInterrupt:
        print("\n\nExperiment paused/stopped by user. Run the script again to continue or inspect results.")
    finally:
        show_results(results)
        print(f"\nExperiment run complete or paused. Full logs are in '{OUTPUT_LOG_DIR}/',")
        print(f"and error-related logs are copied into '{ERROR_LOG_DIR}/'.\n")


if __name__ == "__main__":
    main()

