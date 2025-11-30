import json
import os
import subprocess
import tempfile
from copy import deepcopy

from pysat.examples.rc2 import RC2Stratified
from pysat.formula import WCNF

expl_factor = 1
corr_factor = 1

# Encoding decision diagrams
def phi_E(sat_file,num_of_feature_nodes,feature_partition,label_partition):

    # assign feature nodes to unique features
    def phi_E_feature_det(sat_file,num_of_feature_nodes,feature_partition):
        sat_file.write("phi_E_feature_det :=\n")

        sat_file.write("T")
        for node in range(num_of_feature_nodes):
            sat_file.write(" &\n")
            sat_file.write("(F")
            for feature in feature_partition.keys():
                sat_file.write(" | ")
                sat_file.write("(")
                sat_file.write(f"lam_{node:d}_{feature}")
                for featurep in feature_partition.keys():
                    if featurep!=feature:
                        sat_file.write(" & ")
                        sat_file.write(f"!lam_{node:d}_{featurep}")
                sat_file.write(")")
            sat_file.write(")")
        sat_file.write(";\n")

    # transition relation of template
    def phi_E_transitions(sat_file,num_of_feature_nodes,features_partition,label_partition):
        sat_file.write("phi_E_transitions := \n")

        sat_file.write("T")
        for node in range(num_of_feature_nodes):
            for feature, feature_buckets in features_partition.items():
                if feature_buckets > 0:
                    sat_file.write(" & \n")
                    sat_file.write("(")
                    sat_file.write(f"lam_{node:d}_{feature} =>\n")
                    sat_file.write("          (T")
                    for feature_bucket in range(feature_buckets):
                        sat_file.write(" &\n")
                        sat_file.write("          (F")
                        for nodep in range(node+1,num_of_feature_nodes):
                            sat_file.write(" | ")
                            sat_file.write("(")
                            sat_file.write(f"tau_{node:d}_{feature_bucket:d}_{nodep} &")
                            sat_file.write("(T")
                            for nodepp in range(node+1,num_of_feature_nodes):
                                if nodepp != nodep:
                                    sat_file.write(" & ")
                                    sat_file.write(f"!tau_{node:d}_{feature_bucket:d}_{nodepp}")
                            for label, label_buckets in label_partition.items():
                                for label_bucket in range(label_buckets):
                                    sat_file.write(" & ")
                                    sat_file.write(f"!tau_{node:d}_{feature_bucket:d}_{label}_{label_bucket}")
                            sat_file.write(")")
                            sat_file.write(")")
                        sat_file.write(" | (T")
                        for label, label_buckets in label_partition.items():
                            sat_file.write(" & ")
                            sat_file.write("(F")
                            for label_bucket in range(label_buckets):
                                sat_file.write(" | ")
                                sat_file.write("(")
                                sat_file.write(f"tau_{node:d}_{feature_bucket:d}_{label}_{label_bucket:d} & ")
                                sat_file.write("(T")
                                for label_bucketp in range(label_buckets):
                                    if label_bucketp != label_bucket:
                                        sat_file.write(" & ")
                                        sat_file.write(f"!tau_{node:d}_{feature_bucket:d}_{label}_{label_bucketp}")
                                for nodep in range(node+1,num_of_feature_nodes):
                                    sat_file.write(" & ")
                                    sat_file.write(f"!tau_{node:d}_{feature_bucket:d}_{nodep}")
                                sat_file.write(")")
                                sat_file.write(")")

                            sat_file.write(")")

                        sat_file.write(")")
                        sat_file.write(")")
                    sat_file.write(")")
                    sat_file.write(")")
        sat_file.write(";\n")

    # only buckets of feature in node are allowed
    def phi_E_transitions_consistency(sat_file,num_of_feature_nodes,feature_partition,label_partition):
        sat_file.write("phi_E_transitions_consistency :=\n")

        # compute maximum number of buckets
        max_bucket = 0
        for buckets in feature_partition.values():
            if buckets>max_bucket:
                max_bucket = buckets

        sat_file.write("T")
        for node in range(num_of_feature_nodes):
            for feature, feature_buckets in feature_partition.items():
                sat_file.write(" &\n")
                sat_file.write("(")
                sat_file.write(f"lam_{node:d}_{feature} => ")
                sat_file.write("(T")
                for feature_bucket in range(feature_buckets,max_bucket):
                    for nodep in range(node+1,num_of_feature_nodes):
                        sat_file.write(" & ")
                        sat_file.write(f"!tau_{node}_{feature_bucket:d}_{nodep:d}")
                    for label,label_buckets in label_partition.items():
                        for label_bucket in range(label_buckets):
                            sat_file.write(" & ")
                            sat_file.write(f"!tau_{node}_{feature_bucket:d}_{label}_{label_bucket:d}")
                sat_file.write(")")
                sat_file.write(")")
        sat_file.write(";\n")

    phi_E_feature_det(sat_file,num_of_feature_nodes,feature_partition)
    sat_file.write("\n")
    phi_E_transitions(sat_file,num_of_feature_nodes,feature_partition,label_partition)
    sat_file.write("\n")
    phi_E_transitions_consistency(sat_file,num_of_feature_nodes,feature_partition,label_partition)
    sat_file.write("\n")
    sat_file.write("phi_E := phi_E_feature_det & phi_E_transitions & phi_E_transitions_consistency;\n\n")

# simulate samples
def phi_sim(sat_file,num_of_feature_nodes,feature_partition,label_partition,samples,feature_defs):

    def phi_sim_feature_assignments(sat_file,feature_partition,sample):
        sat_file.write(f"phi_sim_feature_assignments_{sample['id']} :=\n")

        sat_file.write("T")
        sample_assigned_features = feature_partition.keys()
        for feature in sample_assigned_features:
            sat_file.write(" & ")
            sat_file.write("(")
            sat_file.write(f"feature_{sample['id']}_{feature} <=> ")
            sat_file.write(feature_defs[feature][sample['id']])
            sat_file.write(")")
        sat_file.write(";\n")

    def phi_sim_leaf_assignments(sat_file,num_of_feature_nodes,label_partition,sample):
        sat_file.write(f"phi_sim_leaf_assignments_{sample['id']} :=\n")

        sat_file.write("T")
        for label, label_buckets in label_partition.items():
            for label_bucket in range(label_buckets):
                sat_file.write(" & ")
                sat_file.write("(")
                sat_file.write(f"eta_{sample['id']}_{label}_{label_bucket:d} <=> ")
                sat_file.write("(")
                sat_file.write(f"class_{sample['id']}_{label}")
                for label_bucketp in range(label_buckets):
                    if label_bucketp!=label_bucket:
                        sat_file.write(" & ")
                        sat_file.write(f"!class_{sample['id']}_{label_bucketp}")
                sat_file.write(")")
                sat_file.write(")")
        sat_file.write(";\n")

    def phi_sim_tau_unique(sat_file,num_of_feature_nodes,feature_partition,label_partition,sample):
        sat_file.write(f"phi_sim_tau_unique_{sample['id']} :=\n")

        sat_file.write("T")
        for node in range(num_of_feature_nodes):
            sat_file.write(" & ")
            sat_file.write("(")
            sat_file.write(f"kappa_{sample['id']}_{node:d} => ")
            sat_file.write("(T")
            for feature, feature_buckets in feature_partition.items():
                for feature_bucket in range(feature_buckets):
                    sat_file.write(" & ")
                    sat_file.write("(")
                    sat_file.write(f"feature_{sample['id']}_{feature} => ")
                    sat_file.write("(")
                    sat_file.write(f"(lambda_{sample['id']}_{node}_{feature}_{feature_bucket} <=> (")
                    sat_file.write(f"feature_{sample['id']}_{feature}_{feature_bucket} & ")
                    sat_file.write("(T")
                    for label, label_buckets in label_partition.items():
                        for label_bucket in range(label_buckets):
                            sat_file.write(" & ")
                            sat_file.write("(")
                            sat_file.write(f"eta_{sample['id']}_{label}_{label_bucket} => ")
                            sat_file.write("(")
                            sat_file.write("(T")
                            for nodep in range(node+1,num_of_feature_nodes):
                                sat_file.write(" & ")
                                sat_file.write(f"!tau_{node}_{feature_bucket}_{nodep}")
                            sat_file.write(")")
                            sat_file.write(" & ")
                            sat_file.write("(")
                            sat_file.write(f"tau_{node}_{feature_bucket}_{label}_{label_bucket} <=> ")
                            sat_file.write(f"tau_{sample['id']}_{node}_{feature}_{feature_bucket}_{label}_{label_bucket}")
                            sat_file.write(")")
                            sat_file.write(")")
                            sat_file.write(")")
                        sat_file.write(")")
                    sat_file.write(")")
                    sat_file.write(")")
                    sat_file.write(")")
                sat_file.write(")")
            sat_file.write(")")
            sat_file.write(")")
        sat_file.write(";\n")

    def phi_sim_tau_root(sat_file,feature_partition,sample):
        sat_file.write(f"phi_sim_tau_root_{sample['id']} :=\n")

        sat_file.write("T")
        for feature, feature_buckets in feature_partition.items():
            sat_file.write(" & ")
            sat_file.write("(")
            sat_file.write(f"lambda_{sample['id']}_0_{feature}_0 <=> ")
            sat_file.write("(")
            sat_file.write(f"feature_{sample['id']}_{feature}_0 & ")
            sat_file.write("T")
            for label, label_buckets in label_partition.items():
                for label_bucket in range(label_buckets):
                    sat_file.write(" & ")
                    sat_file.write(f"!tau_0_0_{label}_{label_bucket}")
            sat_file.write(")")
            sat_file.write(")")
            sat_file.write(")")
        sat_file.write(";\n")

    def phi_sim_tau_left(sat_file,num_of_feature_nodes,feature_partition,label_partition,sample):
        sat_file.write(f"phi_sim_tau_left_{sample['id']} :=\n")

        sat_file.write("T")
        for node in range(num_of_feature_nodes):
            sat_file.write(" & ")
            sat_file.write("(")
            sat_file.write(f"kappa_{sample['id']}_{node:d} => (T")
            for feature, feature_buckets in feature_partition.items():
                for feature_bucket in range(feature_buckets):
                    sat_file.write(" & ")
                    sat_file.write("(")
                    sat_file.write(f"feature_{sample['id']}_{feature} => ")
                    sat_file.write("(")
                    sat_file.write(f"feature_{sample['id']}_{feature}_{feature_bucket+1} <=> ")
                    sat_file.write("(")
                    sat_file.write(f"lambda_{sample['id']}_{node}_{feature}_{feature_bucket+1} | ")
                    sat_file.write("(T")
                    for nodep in range(node+1,num_of_feature_nodes):
                        sat_file.write(" & ")
                        sat_file.write(f"((tau_{node}_{feature_bucket}_{nodep}) => ")
                        sat_file.write("(")
                        sat_file.write(f"lambda_{sample['id']}_{nodep}_{feature}_{feature_bucket+1}")
                        sat_file.write(")")
                        sat_file.write(")")
                    sat_file.write(")")
                    sat_file.write(")")
                    sat_file.write(")")
                    sat_file.write(")")
                    sat_file.write(")")
            sat_file.write(")")
            sat_file.write(")")
        sat_file.write(";\n")

    def phi_sim_tau_right(sat_file,num_of_feature_nodes,feature_partition,label_partition,sample):
        sat_file.write(f"phi_sim_tau_right_{sample['id']} :=\n")

        sat_file.write("T")
        for node in range(num_of_feature_nodes):
            sat_file.write(" & ")
            sat_file.write("(")
            sat_file.write(f"kappa_{sample['id']}_{node:d} => (T")
            for feature, feature_buckets in feature_partition.items():
                for feature_bucket in range(feature_buckets):
                    sat_file.write(" & ")
                    sat_file.write("(")
                    sat_file.write(f"feature_{sample['id']}_{feature} => ")
                    sat_file.write("(")
                    sat_file.write(f"feature_{sample['id']}_{feature}_{feature_bucket} <=> ")
                    sat_file.write("(")
                    sat_file.write(f"lambda_{sample['id']}_{node}_{feature}_{feature_bucket} | ")
                    sat_file.write("(T")
                    for label, label_buckets in label_partition.items():
                        for label_bucket in range(label_buckets):
                            sat_file.write(" & ")
                            sat_file.write("(")
                            sat_file.write(f"((tau_{node}_{feature_bucket}_{label}_{label_bucket})) => ")
                            sat_file.write("(")
                            sat_file.write(f"eta_{sample['id']}_{label}_{label_bucket}")
                            sat_file.write(")")
                            sat_file.write(")")
                    sat_file.write(")")
                    sat_file.write(")")
                    sat_file.write(")")
                    sat_file.write(")")
                    sat_file.write(")")
            sat_file.write(")")
            sat_file.write(")")
        sat_file.write(";\n")

    def phi_sim_feature_cardinality(sat_file,feature_partition,sample):
        sat_file.write(f"phi_sim_feature_cardinality_{sample['id']} :=\n")

        sat_file.write("T")
        for feature, feature_buckets in feature_partition.items():
            sat_file.write(" & ")
            sat_file.write("(")
            sat_file.write(f"(feature_{sample['id']}_{feature}_0 <=> ")
            sat_file.write(f"lambda_{sample['id']}_0_{feature}_0)\n")
            sat_file.write(" & ")
            sat_file.write("(")
            sat_file.write(f"feature_{sample['id']}_{feature}_1 <=> ")
            sat_file.write("(")
            sat_file.write(f"lambda_{sample['id']}_0_{feature}_1 | ")
            sat_file.write("lambda_{sample['id']}_1_{feature}_1)\n")
            sat_file.write(")")
            for bucket in range(2,feature_buckets):
                sat_file.write(" & ")
                sat_file.write("(")
                sat_file.write(f"feature_{sample['id']}_{feature}_{bucket:d} <=> ")
                sat_file.write("(")
                sat_file.write(" | ".join(f"lambda_{sample['id']}_{bucketp}_{feature}_{bucket}"
                                           for bucketp in range(bucket + 1)))
                sat_file.write(")")
                sat_file.write(")\n")
            sat_file.write(")")
        sat_file.write(";\n")

    for sample in samples:
        phi_sim_feature_assignments(sat_file,feature_partition,sample)
        sat_file.write("\n")
        phi_sim_leaf_assignments(sat_file,num_of_feature_nodes,label_partition,sample)
        sat_file.write("\n")
        phi_sim_tau_unique(sat_file,num_of_feature_nodes,feature_partition,label_partition,sample)
        sat_file.write("\n")
        phi_sim_tau_root(sat_file,feature_partition,sample)
        sat_file.write("\n")
        phi_sim_tau_left(sat_file,num_of_feature_nodes,feature_partition,label_partition,sample)
        sat_file.write("\n")
        phi_sim_tau_right(sat_file,num_of_feature_nodes,feature_partition,label_partition,sample)
        sat_file.write("\n")
        phi_sim_feature_cardinality(sat_file,feature_partition,sample)
        sat_file.write("\n")

    sat_file.write("phi_sim := \n")
    sat_file.write("(T\n")
    for sample in samples:
        sat_file.write(f"   & phi_sim_feature_assignments_{sample['id']}\n")
        sat_file.write(f"   & phi_sim_leaf_assignments_{sample['id']}\n")
        sat_file.write(f"   & phi_sim_tau_unique_{sample['id']}\n")
        sat_file.write(f"   & phi_sim_tau_root_{sample['id']}\n")
        sat_file.write(f"   & phi_sim_tau_left_{sample['id']}\n")
        sat_file.write(f"   & phi_sim_tau_right_{sample['id']}\n")
        sat_file.write(f"   & phi_sim_feature_cardinality_{sample['id']}\n")
    sat_file.write(");\n\n")

# encode correlation
def phi_corr(sat_file,samples):

    def phi_corr_kappa_root(sat_file,samples):
        sat_file.write("phi_corr_kappa_root :=\n")

        sat_file.write("T")
        for sample in samples:
            sat_file.write(" & ")
            sat_file.write("(")
            sat_file.write(f"kappa_{sample['id']}_0")
            sat_file.write(")")
        sat_file.write(";\n")

    def phi_corr_kappa_unique(sat_file,samples):
        sat_file.write("phi_corr_kappa_unique :=\n")

        sat_file.write("T")
        for node in range(len(samples)):
            sat_file.write(" & ")
            sat_file.write("(")
            sat_file.write(f"kappa_{samples[node]['id']}_{node:d} => ")
            sat_file.write("(T")
            for nodep in range(len(samples)):
                if node != nodep:
                    sat_file.write(" & ")
                    sat_file.write(f"!kappa_{samples[node]['id']}_{nodep}")
            sat_file.write(")")
            sat_file.write(")")
        sat_file.write(";\n")

    def phi_corr_kappa_leaf(sat_file,samples):
        sat_file.write("phi_corr_kappa_leaf :=\n")

        sat_file.write("T")
        for node in range(len(samples)):
            sat_file.write(" & ")
            sat_file.write("(")
            sat_file.write(f"kappa_{samples[node]['id']}_{node:d} => ")
            sat_file.write("(")
            sat_file.write(" | ".join(f"class_{samples[node]['id']}_{label}"
                                      for label in range(len(samples[node]['class']))))
            sat_file.write(")")
            sat_file.write(")")
        sat_file.write(";\n")

    phi_corr_kappa_root(sat_file,samples)
    sat_file.write("\n")
    phi_corr_kappa_unique(sat_file,samples)
    sat_file.write("\n")
    phi_corr_kappa_leaf(sat_file,samples)
    sat_file.write("\n")
    sat_file.write("phi_corr := phi_corr_kappa_root & phi_corr_kappa_unique & phi_corr_kappa_leaf;\n\n")

# encode explanations
def phi_expl(sat_file,num_of_feature_nodes,feature_partition,label_partition):

    def phi_expl_used_nodes(sat_file,num_of_feature_nodes):
        sat_file.write("phi_expl_used_nodes :=\n")

        sat_file.write("T")
        for node in range(num_of_feature_nodes):
            sat_file.write(" & ")
            sat_file.write("(")
            sat_file.write(f"used_node_{node:d} <=> ")
            sat_file.write("(")
            sat_file.write("(")
            sat_file.write(f"lam_{node:d}_0")
            sat_file.write(")")
            for feature, feature_buckets in feature_partition.items():
                for feature_bucket in range(feature_buckets):
                    sat_file.write(" | ")
                    sat_file.write("(")
                    sat_file.write(f"tau_{node:d}_{feature_bucket:d}_{feature}")
                    sat_file.write(")")
            sat_file.write(")")
            sat_file.write(")")
        sat_file.write(";\n")

    def phi_expl_used_nodes_ordering(sat_file,num_of_feature_nodes):
        sat_file.write("phi_expl_used_nodes_ordering :=\n")

        sat_file.write("T")
        for node in range(num_of_feature_nodes - 1):
            sat_file.write(" & ")
            sat_file.write("(")
            sat_file.write(f"used_node_{node+1:d} => used_node_{node:d}")
            sat_file.write(")")
        sat_file.write(";\n")

    phi_expl_used_nodes(sat_file,num_of_feature_nodes)
    sat_file.write("\n")
    phi_expl_used_nodes_ordering(sat_file,num_of_feature_nodes)
    sat_file.write("\n")
    sat_file.write("phi_expl := phi_expl_used_nodes & phi_expl_used_nodes_ordering;\n\n")

def phi_region(
    sat_file,
    num_of_feature_nodes,
    feature_partition,
    feature_weights,
    lower_bound,
    upper_bound,
    precision,
    experiment_dir=None,
):

    def phi_exp_weights(sat_file,num_of_feature_nodes,feature_partition,weights_map,precision):
        sat_file.write("phi_exp_weights := \n")

        sat_file.write("T")
        for weight, weight_value in weights_map.items():
            sat_file.write(" & ")
            sat_file.write("(")
            sat_file.write(f"{weight} => ")
            sat_file.write("(")
            sat_file.write(enc_in_binary(weight_value, f"{weight}", precision))
            sat_file.write(")")
            sat_file.write(")")
        sat_file.write(";\n")

    def phi_adder(sat_file,num_of_feature_nodes,precision):
        sat_file.write("phi_adder := \n")

        sat_file.write("T")
        for node in range(num_of_feature_nodes):
            sat_file.write(" & ")
            sat_file.write("(")
            sat_file.write(f"adder_{node:d}_0 == ")
            sat_file.write("(")
            sat_file.write(f"used_node_{node:d} & explainability_{node:d}_0")
            sat_file.write(")")
            sat_file.write(")")
        sat_file.write(";\n")

    def phi_threshold(sat_file,lower_bound,upper_bound,precision, lower, upper):
        sat_file.write(f"phi_threshold := T ")

        if upper:
            sat_file.write("&\n")
            sat_file.write(f"smaller_{precision-1}")
            for bit in range(1,precision):
                sat_file.write(f"& (smaller_{bit} == ((!a_fin_{bit} & upper_{bit})| ( (a_fin_{bit} == upper_{bit}) & smaller_{bit-1})))")
            sat_file.write(f"& (smaller_0 == (!a_fin_0 & upper_0))\n")

        if lower:
            sat_file.write("&\n")
            sat_file.write(f"larger_{precision-1}")
            for bit in range(1,precision):
                sat_file.write(f"& (larger_{bit} == ((a_fin_{bit} & !lower_{bit})| ((a_fin_{bit} == lower_{bit}) & larger_{bit-1})))")
            sat_file.write(f"& (larger_0 == (a_fin_0 & !lower_0))\n")

        if upper:
            sat_file.write("& \n")
            temp = enc_in_binary(upper_bound, "upper", precision)
            sat_file.write(temp)
        if lower:
            sat_file.write("& \n")
            temp = enc_in_binary(lower_bound, "lower", precision)
            sat_file.write(temp)
        sat_file.write(";\n\n")

    if lower_bound>0 or upper_bound<100:
        # Encode weights of syntactic structures
        weights_map = compute_normalizied_weights(
            feature_weights,
            num_of_feature_nodes,
            feature_partition,
            experiment_dir,
        )
        phi_exp_weights(sat_file,num_of_feature_nodes,feature_partition,weights_map,precision)
        # Encode adder for summing up weights
        phi_adder(sat_file,num_of_feature_nodes,precision)
        phi_threshold(sat_file,lower_bound,upper_bound,precision, lower_bound>0, upper_bound<100)
        sat_file.write("phi_region := phi_exp_weights & phi_adder & phi_threshold;\n\n ")
    else:
        sat_file.write("phi_region := T;\n\n")

def enc_in_binary(number, prefix, precision):
    ret = f"({prefix}_{precision-1} == {int((number/(2**(precision-1)))%2)})"
    for bit in reversed(range(precision-1)):
        ret += f"\n& ({prefix}_{bit} == {int((number/(2**bit))%2)})"

    return ret

def compute_normalizied_weights(
    feature_weights,
    num_of_feature_nodes,
    feature_partition,
    experiment_dir=None,
):
    max_weight, weights_map = compute_or_load_max_weight(
        feature_weights,
        num_of_feature_nodes,
        feature_partition,
        experiment_dir,
    )

    normalized_map = deepcopy(weights_map)
    for key in normalized_map:
        if max_weight == 0:
            normalized_map[key] = 0
        else:
            normalized_map[key] = int((normalized_map[key] / max_weight) * 100)

    return normalized_map

def compute_or_load_max_weight(
    feature_weights,
    num_of_feature_nodes,
    feature_partition,
    experiment_dir=None,
):
    cache_key = (
        tuple(sorted(feature_weights.items())),
        num_of_feature_nodes,
        tuple(sorted(feature_partition.items())),
    )

    if not hasattr(compute_or_load_max_weight, "_memory_cache"):
        compute_or_load_max_weight._memory_cache = {}
    memory_cache = compute_or_load_max_weight._memory_cache

    if cache_key in memory_cache:
        return memory_cache[cache_key]

    if experiment_dir is not None:
        cache_path = os.path.join(experiment_dir, "explainability_weights.json")
        if os.path.isfile(cache_path):
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                cached_entries = json.load(cache_file)
            serialized_key = serialize_cache_key(cache_key)
            if serialized_key in cached_entries:
                entry = cached_entries[serialized_key]
                max_weight = entry["max_weight"]
                weights_map = entry["weights_map"]
                memory_cache[cache_key] = (max_weight, weights_map)
                return max_weight, weights_map

    max_weight, weights_map = compute_max_explainability_weight(
        feature_weights,
        num_of_feature_nodes,
        feature_partition,
    )

    memory_cache[cache_key] = (max_weight, weights_map)

    if experiment_dir is not None:
        os.makedirs(experiment_dir, exist_ok=True)
        cache_path = os.path.join(experiment_dir, "explainability_weights.json")
        if os.path.isfile(cache_path):
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                cached_entries = json.load(cache_file)
        else:
            cached_entries = {}
        serialized_key = serialize_cache_key(cache_key)
        cached_entries[serialized_key] = {
            "max_weight": max_weight,
            "weights_map": weights_map,
        }
        with open(cache_path, "w", encoding="utf-8") as cache_file:
            json.dump(cached_entries, cache_file, indent=2, sort_keys=True)

    return max_weight, weights_map

def serialize_cache_key(cache_key):
    feature_weights_items, num_of_feature_nodes, feature_partition_items = cache_key

    return json.dumps(
        {
            "feature_weights": list(feature_weights_items),
            "num_of_feature_nodes": num_of_feature_nodes,
            "feature_partition": list(feature_partition_items),
        },
        sort_keys=True,
    )

def compute_max_explainability_weight(
    feature_weights,
    num_of_feature_nodes,
    feature_partition,
):
    wcnf = build_phi_expl_wcnf(
        feature_weights,
        num_of_feature_nodes,
        feature_partition,
    )

    with RC2Stratified(wcnf) as rc2:
        max_weight = rc2.compute()
        model = rc2.model

    weights_map = extract_weight_assignments(
        model,
        wcnf,
    )

    return max_weight, weights_map

def build_phi_expl_wcnf(
    feature_weights,
    num_of_feature_nodes,
    feature_partition,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        bc_path = os.path.join(tmpdir, "phi_expl.bc")
        dimacs_path = os.path.join(tmpdir, "phi_expl.dimacs")
        weights_path = os.path.join(tmpdir, "phi_expl.weights.json")

        with open(bc_path, "w", encoding="utf-8") as sat_file:
            sat_file.write("BC1.1\n")
            phi_expl(sat_file, num_of_feature_nodes, feature_partition, {})
            sat_file.write("FORMULA := phi_expl;\n")
            sat_file.write("ASSIGN FORMULA;\n")

        subprocess.run(
            [
                "./synthesizer/max_sat/bc2cnf",
                "-all",
                bc_path,
                dimacs_path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        soft_vars, _, _, _ = extract_soft_variables(
            dimacs_path,
            feature_weights,
            num_of_feature_nodes,
            feature_partition,
            experiment_dir=None,
        )

        weights_map = {
            key: value
            for key, value in feature_weights.items()
        }

        with open(weights_path, "w", encoding="utf-8") as weights_file:
            json.dump(weights_map, weights_file)

        wcnf = WCNF()
        with open(dimacs_path, "r", encoding="utf-8") as dimacs_file:
            for line in dimacs_file:
                if line.startswith("p"):
                    _, _, num_vars, num_clauses = line.split()
                    wcnf.nv = int(num_vars)
                elif line.startswith("c"):
                    continue
                else:
                    clause = list(map(int, line.split()))
                    clause.pop()
                    wcnf.append(clause, weight=wcnf.top)
        for var, weight in soft_vars:
            wcnf.append([int(var)], weight=weight)

        return wcnf

def extract_weight_assignments(model, wcnf):
    weights_map = {}
    for idx, weight in enumerate(wcnf.wght):
        if weight == wcnf.top:
            continue
        literal = wcnf.clauses[idx][0]
        var_name = f"used_node_{abs(literal) - 1}"
        if (literal > 0 and model[abs(literal) - 1]) or (literal < 0 and not model[abs(literal) - 1]):
            weights_map[var_name] = weight
    return weights_map

def extract_threshold_vars(dimacs):
    print("|---Extracting threshold variables...")

    threshold_vars = {}
    text_file = open(dimacs)

    for line in text_file.readlines():
        if not line.startswith("c"):
            break

        current_num = ""
        current_var = ""

        if ("upper_" in line or "lower_" in line):
            words = line.split()
            current_var = words[1]
            current_num = words[3]

            threshold_vars[current_var]= current_num

    return threshold_vars

def extract_soft_variables(dimacs,feature_weights,num_of_feature_nodes, feature_partition, experiment_dir=None):
    print("|---Extracting soft variables...")
    soft_vars = []
    expl_soft_vars = []
    corr_soft_vars = []
    corr_vars = []
    text_file = open(dimacs)


    weights_map = compute_normalizied_weights(feature_weights,num_of_feature_nodes, feature_partition, experiment_dir)

    for line in text_file.readlines():
        if not line.startswith("c"):
            break

        current_num = ""
        current_var = ""
        if ("match_0" in line or " lamp_" in line or "used_node_" in line):
            words = line.split()
            current_var = words[1]
            current_num = words[3]

            if current_var in weights_map.keys():
                weight = int(weights_map[current_var]/corr_factor)
                if "used_node_" in current_var:
                    if((f"-{current_num}",weight) not in soft_vars):
                        soft_vars.append((f"-{current_num}",weight))
                        expl_soft_vars.append((f"-{current_num}",weight))
                if "lamp_" in current_var:
                    if((current_num,weight) not in soft_vars):
                        soft_vars.append((current_num,weight))
                        expl_soft_vars.append((current_num,weight))
            else:
                weight = 1
                if((current_num,weight) not in soft_vars):
                        soft_vars.append((current_num,weight))
                        corr_soft_vars.append((current_num,weight))
                corr_vars.append((current_num,weight))


            # reached_num = False
            # finished_var = False
            # for letter in line:
            #     if letter=='\n':
            #         break
            #     if reached_num:
            #         if letter.isdigit():
            #             current_num += letter
            #     if letter == '>':
            #         reached_num = True

    text_file.close()
    return soft_vars, corr_soft_vars, expl_soft_vars, corr_vars
# def phi_soft(sat_file, samples, num_of_feature_nodes,feature_partition):
#     sat_file.write("phi_soft := ")
#     phi_corr(sat_file,samples)
#     for node in range(num_of_feature_nodes):
#         sat_file.write(f"& !used_node_{node:d}")
#     for node in range(num_of_feature_nodes):
#         for feature, buckets in feature_partition.items():
#             sat_file.write(f" & lam_{node:d}_{feature}")
#     sat_file.write(";\n\n")
