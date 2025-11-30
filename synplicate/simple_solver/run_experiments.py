import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
BASE_PATH = REPO_ROOT / "experiments"
CHECKPOINT_FILE = BASE_DIR / "experiment_results.json"
OUTPUT_LOG_DIR = BASE_DIR / "term_output"
ERROR_LOG_DIR = OUTPUT_LOG_DIR / "errors"
SNAPSHOT_DIR_NAME = "previous_runs"
VIEW = "linear"
TIMEOUT_SECONDS = 600


def discover_experiments() -> List[str]:
    experiments = []
    for root, _, files in os.walk(BASE_PATH):
        if "dd.config" in files:
            exp_path = Path(root)
            experiments.append(str(exp_path.relative_to(BASE_PATH)))
    return sorted(experiments)


def parse_output(output_string: str):
    import re

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


def display_linear(results: Dict[str, Dict[str, str]], current_exp=None, elapsed=None):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("--- Simple Solver Experiment Runner (Linear View) ---\n")

    for exp_name in results:
        print(f"Experiment: {exp_name}")
        res = results.get(exp_name, "Not Run")
        if isinstance(res, dict) and {"time", "acc", "exp", "cost", "corR", "expR"}.issubset(res.keys()):
            line = (
                f"time={res['time']:.2f}s, cost={int(res['cost'])}, "
                f"acc={res['acc']:.2f}",
                # f"acc={res['acc']:.2f}, exp={res['exp']:.2f}, "
                # f"corR={res['corR']}, expR={res['expR']}"
            )
        else:
            line = "Not Run" if isinstance(res, dict) else str(res)
        print(f"  - simple_solver: {line}\n")

    if current_exp is not None and elapsed is not None:
        print(f"Currently running: {current_exp} | Elapsed: {int(elapsed)}s")

    print("\nPress Ctrl+C while a run is active to SKIP that experiment;")
    print("press Ctrl+C between runs to pause/stop the whole script.")
    print("-" * 30)


def save_results(results):
    with CHECKPOINT_FILE.open('w') as f:
        json.dump(results, f, indent=4)


def load_results():
    if CHECKPOINT_FILE.exists():
        try:
            with CHECKPOINT_FILE.open('r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def show_results(results, current_exp=None, elapsed=None):
    display_linear(results, current_exp=current_exp, elapsed=elapsed)


def copy_error_log(log_filename: Path):
    try:
        ERROR_LOG_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(log_filename, ERROR_LOG_DIR / log_filename.name)
    except Exception:
        pass


def snapshot_program(exp_dir: Path, solver_tag: str):
    try:
        program_dir = exp_dir / "program"
        if program_dir.is_dir():
            snapshot_root = exp_dir / SNAPSHOT_DIR_NAME
            snapshot_root.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            snapshot_dir = snapshot_root / f"program_{solver_tag}_{timestamp}"
            shutil.copytree(program_dir, snapshot_dir)
    except Exception:
        pass


def run_single_experiment(exp_name: str, results: Dict[str, Dict[str, str]]):
    existing = results.get(exp_name)
    if isinstance(existing, dict) and {"time", "acc", "exp", "cost", "corR", "expR"}.issubset(existing.keys()):
        return results

    results[exp_name] = "Running..."
    show_results(results, current_exp=exp_name, elapsed=0)

    exp_path = BASE_PATH / exp_name
    command = ["python3", str(BASE_DIR / "simple_solver.py"), str(exp_path) + "/"]

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

    try:
        while True:
            if process.poll() is not None:
                break

            elapsed = time.time() - start_time

            if TIMEOUT_SECONDS and elapsed > TIMEOUT_SECONDS:
                timed_out = True
                process.kill()
                break

            show_results(results, current_exp=exp_name, elapsed=elapsed)
            print(f"Executing: {' '.join(command)}")
            time.sleep(1)

    except KeyboardInterrupt:
        skipped = True
        try:
            process.kill()
        except Exception:
            pass

    stdout, stderr = process.communicate()

    safe_exp_name = exp_name.replace('/', '_')
    log_filename = OUTPUT_LOG_DIR / f"{safe_exp_name}_simple_solver.log"
    OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with log_filename.open('w') as log_file:
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

    if skipped:
        results[exp_name] = "Skipped by user"
        save_results(results)
        copy_error_log(log_filename)
        show_results(results)
        return results

    if timed_out:
        results[exp_name] = f"Timeout after {int(elapsed)}s"
        save_results(results)
        copy_error_log(log_filename)
        show_results(results)
        return results

    if process.returncode == 0:
        parsed_data = parse_output(stdout)
        if parsed_data:
            results[exp_name] = parsed_data
        else:
            results[exp_name] = "Parse Error"
            copy_error_log(log_filename)
    else:
        results[exp_name] = f"Run Error (Code: {process.returncode})"
        copy_error_log(log_filename)

    save_results(results)
    snapshot_program(BASE_PATH / exp_name, "simple_solver")
    show_results(results)
    return results


def main():
    global OUTPUT_LOG_DIR, ERROR_LOG_DIR

    OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_LOG_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()

    experiments = discover_experiments()

    if results:
        show_results(results)
        try:
            choice = input("Previous results found. [C]ontinue or [S]tart fresh? ").strip().lower()
        except EOFError:
            choice = 'c'
        if choice.startswith('s'):
            print("Starting fresh, clearing previous results...")
            results = {}
            time.sleep(1)

    for exp_name in experiments:
        if exp_name not in results:
            results[exp_name] = "Not Run"

    try:
        for exp_name in experiments:
            results = run_single_experiment(exp_name, results)
    except KeyboardInterrupt:
        print("\n\nExperiment paused/stopped by user. Run the script again to continue or inspect results.")
    finally:
        show_results(results)
        print(f"\nExperiment run complete or paused. Full logs are in '{OUTPUT_LOG_DIR}/',")
        print(f"and error-related logs are copied into '{ERROR_LOG_DIR}/'.\n")


if __name__ == "__main__":
    main()
