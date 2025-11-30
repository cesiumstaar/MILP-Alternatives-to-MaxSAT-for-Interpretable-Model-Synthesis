# sampler.py for the 4-feature Iris dataset (On-Disk Caching Version)

import os
import pickle
import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# --- Module-level Configuration ---
_CACHE_FILENAME = "sampler_cache_4_features.pkl"
_cached_data = {
    "samples": {},
    "model": None
}
_is_cache_loaded = False
_feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
_iris_data_4_features = None

def _load_or_initialize_cache():
    """
    Loads data from the cache file if it exists. Otherwise, it generates
    the initial data, trains a model, and creates the cache file.
    """
    global _cached_data, _is_cache_loaded, _iris_data_4_features

    if os.path.exists(_CACHE_FILENAME):
        print(f"--- Found cache file '{_CACHE_FILENAME}'. Loading data. ---")
        with open(_CACHE_FILENAME, 'rb') as f:
            _cached_data = pickle.load(f)
        # We need to re-populate the feature data for min/max calculations
        iris = load_iris()
        _iris_data_4_features = iris.data
            
    else:
        print(f"--- No cache file found. Initializing model and real samples. ---")
        iris = load_iris()
        _iris_data_4_features = iris.data # Use all 4 features
        
        model = DecisionTreeClassifier()
        model.fit(_iris_data_4_features, iris.target)
        _cached_data["model"] = model

        label_name = 'species'
        for i in range(len(_iris_data_4_features)):
            feature_values = _iris_data_4_features[i]
            label_value = iris.target[i]
            sample_key = tuple((name, val) for name, val in zip(_feature_names, feature_values))
            sample_value = [(label_name, int(label_value))]
            _cached_data["samples"][sample_key] = sample_value
        
        with open(_CACHE_FILENAME, 'wb') as f:
            pickle.dump(_cached_data, f)
        print(f"--- Saved initial {len(_cached_data['samples'])} samples to '{_CACHE_FILENAME}'. ---")
        
    _is_cache_loaded = True

def uniform(num_of_samples):
    """
    Provides samples from the on-disk cache. If more samples are requested
    than exist, it generates new synthetic ones and updates the cache file.
    """
    global _cached_data

    if not _is_cache_loaded:
        _load_or_initialize_cache()
    
    if num_of_samples > len(_cached_data["samples"]):
        num_synthetic_needed = num_of_samples - len(_cached_data["samples"])
        print(f"\n--- Cache has {len(_cached_data['samples'])} samples. Requested {num_of_samples}. Generating {num_synthetic_needed} new synthetic samples. ---\n")

        min_vals = _iris_data_4_features.min(axis=0)
        max_vals = _iris_data_4_features.max(axis=0)
        
        for _ in range(num_synthetic_needed):
            synth_features = [random.uniform(min_vals[i], max_vals[i]) for i in range(len(_feature_names))]
            synth_label = _cached_data["model"].predict([synth_features])[0]
            
            sample_key = tuple((name, val) for name, val in zip(_feature_names, synth_features))
            sample_value = [('species', int(synth_label))]
            
            while sample_key in _cached_data["samples"]:
                synth_features = [random.uniform(min_vals[i], max_vals[i]) for i in range(len(_feature_names))]
                sample_key = tuple((name, val) for name, val in zip(_feature_names, synth_features))
            
            _cached_data["samples"][sample_key] = sample_value

        with open(_CACHE_FILENAME, 'wb') as f:
            pickle.dump(_cached_data, f)
        print(f"--- Cache updated. Saved {len(_cached_data['samples'])} total samples to '{_CACHE_FILENAME}'. ---")

    all_samples_list = list(_cached_data["samples"].items())
    random.shuffle(all_samples_list)
    
    return dict(all_samples_list[:num_of_samples])