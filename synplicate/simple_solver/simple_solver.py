import argparse
import importlib
import importlib.util
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_config(experiment_path: Path):
    config_path = experiment_path / "dd.config"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing dd.config in {experiment_path}")

    feature_weights: Dict[str, float] = {}
    feature_defs_module = None
    label_name = None

    with config_path.open("r") as config_file:
        for line in config_file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if stripped.startswith("features"):
                rhs = stripped.split("=", 1)[1]
                for entry in rhs.split(","):
                    entry = entry.strip()
                    if not entry:
                        continue
                    parts = entry.split(":")
                    if len(parts) >= 3:
                        name = parts[0]
                        weight = float(parts[2])
                        feature_weights[name] = weight
            elif stripped.startswith("labels"):
                rhs = stripped.split("=", 1)[1]
                first_label = rhs.split(",")[0].strip()
                label_name = first_label.split(":")[0]
            elif stripped.startswith("feature_defs"):
                rhs = stripped.split("=", 1)[1].strip()
                feature_defs_module = rhs

    if not feature_weights:
        raise ValueError(f"No features found in {config_path}")
    if feature_defs_module is None:
        raise ValueError(f"No feature_defs module found in {config_path}")
    if label_name is None:
        raise ValueError(f"No label name found in {config_path}")

    return feature_weights, feature_defs_module, label_name


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_feature_defs(experiment_path: Path, module_name: str):
    return _load_module(experiment_path / f"{module_name}.py", module_name)


def load_sampler(experiment_path: Path):
    return _load_module(experiment_path / "sampler.py", f"sampler_{experiment_path.name}")


def ensure_cache_loaded(sampler_module: Any):
    load_func = getattr(sampler_module, "_load_or_initialize_cache", None)
    is_loaded = getattr(sampler_module, "_is_cache_loaded", True)
    if callable(load_func) and not is_loaded:
        load_func()


def load_samples(sampler_module: Any, sample_limit: int = None) -> Dict[Tuple[Tuple[str, Any], ...], List[Tuple[str, Any]]]:
    ensure_cache_loaded(sampler_module)

    cached_data = getattr(sampler_module, "_cached_data", None)
    samples: Dict[Tuple[Tuple[str, Any], ...], List[Tuple[str, Any]]] = {}
    if isinstance(cached_data, dict):
        maybe_samples = cached_data.get("samples")
        if isinstance(maybe_samples, dict) and maybe_samples:
            samples = dict(maybe_samples)

    if not samples and hasattr(sampler_module, "uniform"):
        default_request = sample_limit or 100
        samples = sampler_module.uniform(default_request)

    if sample_limit is not None and len(samples) > sample_limit:
        items = list(samples.items())[:sample_limit]
        samples = dict(items)

    return samples


def sanitize_label_value(value: Any) -> str:
    return str(value).replace(" ", "_").replace("/", "_")


def evaluate_feature(samples: Dict[Tuple[Tuple[str, Any], ...], List[Tuple[str, Any]]], feature_name: str, feature_func):
    partition_label_counts: Dict[Any, Counter] = defaultdict(Counter)
    overall_labels = Counter()

    for inputs, outputs in samples.items():
        inputs_list = list(inputs)
        label_value = outputs[0][1] if outputs else None
        partition_value = feature_func(inputs_list)
        partition_label_counts[partition_value][label_value] += 1
        overall_labels[label_value] += 1

    correct_predictions = sum(count.most_common(1)[0][1] for count in partition_label_counts.values())
    total_samples = len(samples)
    accuracy = correct_predictions / total_samples if total_samples else 0.0

    partition_best_labels = {partition: counts.most_common(1)[0][0] for partition, counts in partition_label_counts.items()}
    fallback_label = overall_labels.most_common(1)[0][0] if overall_labels else None

    return accuracy, correct_predictions, partition_best_labels, fallback_label


def build_program_files(
    experiment_path: Path,
    feature_name: str,
    label_name: str,
    partition_map: Dict[Any, Any],
    fallback_label: Any,
    image_dir: Path,
):
    program_dir = experiment_path / "program"
    if program_dir.exists():
        shutil.rmtree(program_dir)
    program_dir.mkdir(parents=True, exist_ok=True)

    python_path = program_dir / f"program_simple_{feature_name}.py"
    dot_path = program_dir / f"program_simple_{feature_name}.dot"

    image_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    png_path = image_dir / f"{experiment_path.name}_program_simple_{feature_name}_{timestamp}.png"

    with python_path.open("w") as program_file:
        program_file.write("import sys\n")
        program_file.write(f"sys.path.insert(0, '{experiment_path}/')\n")
        program_file.write("import feature_defs\n\n")

        program_file.write("def execute(inputs):\n")
        program_file.write("    features = feature_defs.retrieve_feature_defs()\n")
        program_file.write(f"    value = features['{feature_name}'](inputs)\n")

        for partition, label in partition_map.items():
            program_file.write(f"    if value == {repr(partition)}:\n")
            program_file.write(f"        return '{label_name}_{sanitize_label_value(label)}'\n")

        if fallback_label is not None:
            program_file.write("    return '%s'\n" % f"{label_name}_{sanitize_label_value(fallback_label)}")
        else:
            program_file.write("    return ''\n")

    with dot_path.open("w") as dot_file:
        dot_file.write("digraph {\n")
        dot_file.write(f"node [label={feature_name},shape=\"diamond\",style=\"\"] 0\n")
        for partition, label in partition_map.items():
            node_label = f"{label_name}_{sanitize_label_value(label)}"
            dot_file.write(f"node [label={node_label},style=\"\"] {node_label}\n")
            dot_file.write(f"0 -> {node_label} [label=\"{partition}\"]\n")
        if fallback_label is not None:
            fallback_node = f"{label_name}_{sanitize_label_value(fallback_label)}"
            dot_file.write(f"node [label={fallback_node},style=\"\"] {fallback_node}\n")
            dot_file.write(f"0 -> {fallback_node} [label=\"default\"]\n")
        dot_file.write("}")

    dot_status = os.system(f"dot -Tpng {dot_path} -o {png_path}")
    if dot_status != 0:
        print("Warning: graphviz 'dot' command not found or failed; PNG not generated.")
        png_path = None

    return python_path, dot_path, png_path


def main():
    parser = argparse.ArgumentParser(description="Simple single-node solver")
    parser.add_argument("experiment_path", help="Path to the experiment directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on the number of samples to evaluate")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent

    raw_exp_path = Path(args.experiment_path)
    if raw_exp_path.is_absolute():
        exp_path = raw_exp_path.resolve()
    else:
        candidate = (repo_root / raw_exp_path).resolve()
        if candidate.exists():
            exp_path = candidate
        else:
            exp_path = (repo_root.parent / raw_exp_path).resolve()

    os.chdir(repo_root)
    start_time = time.time()

    feature_weights, feature_defs_module, label_name = parse_config(exp_path)

    sampler = load_sampler(exp_path)
    samples = load_samples(sampler, sample_limit=args.max_samples)
    total_samples = len(samples)

    if total_samples == 0:
        print("No samples available for evaluation.")
        return

    feature_defs = load_feature_defs(exp_path, feature_defs_module).retrieve_feature_defs()
    max_weight = max(feature_weights.values()) if feature_weights else 1.0

    best_feature = None
    best_accuracy = 0.0
    best_correct = 0
    best_explainability_score = 0.0
    best_partition_map = {}
    best_fallback_label = None

    for feature, weight in feature_weights.items():
        if feature not in feature_defs:
            continue
        accuracy, correct, partition_map, fallback_label = evaluate_feature(samples, feature, feature_defs[feature])
        explainability_score = weight / max_weight if max_weight else 0.0
        if accuracy + explainability_score > best_accuracy + best_explainability_score:
            best_feature = feature
            best_accuracy = accuracy
            best_correct = correct
            best_explainability_score = explainability_score
            best_partition_map = partition_map
            best_fallback_label = fallback_label

    if best_feature is None:
        print("No suitable feature found for synthesis.")
        return

    correctness_reward = best_correct
    explainibility_reward = int(best_explainability_score * total_samples)
    total_cost = max(0.0, (2 - (best_accuracy + best_explainability_score)) * total_samples)

    image_dir = base_dir / "images"
    python_path, dot_path, png_path = build_program_files(
        exp_path,
        best_feature,
        label_name,
        best_partition_map,
        best_fallback_label,
        image_dir,
    )

    elapsed = time.time() - start_time

    print(f"Explored {len(feature_weights)} features to find a single-node classifier.")
    print(f"Best feature: {best_feature}")
    print(f"Synthesized program: {python_path}")
    print(f"Visualization: {dot_path}")
    if png_path:
        print(f"Image stored at: {png_path}")
    print(f"Synthesis time:{elapsed}")
    print(f"Accuracy:{best_accuracy} ({best_correct}/{total_samples})")
    print(f"Explainability: {best_explainability_score}")
    print(f"Total cost:{total_cost}")
    print(f"correctnessReward: {correctness_reward}")
    print(f"explainibilityReward: {explainibility_reward}")

    summary = {
        "feature": best_feature,
        "accuracy": best_accuracy,
        "explainability": best_explainability_score,
        "correct": best_correct,
        "samples": total_samples,
        "cost": total_cost,
        "correctnessReward": correctness_reward,
        "explainibilityReward": explainibility_reward,
        "program": str(python_path),
        "dot": str(dot_path),
        "png": str(png_path) if png_path else None,
    }

    with (exp_path / "program" / "simple_solver_summary.json").open("w") as summary_file:
        json.dump(summary, summary_file, indent=2)


if __name__ == "__main__":
    main()
