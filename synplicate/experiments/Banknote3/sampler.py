# sampler.py for the 3-feature Banknote dataset (On-Disk Caching Version)

import os
import pickle
import random
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# --- Module-level Configuration ---
_DATA_FILE_PATH = "experiments/Banknote3/data_banknote_authentication.txt"
_CACHE_FILENAME = "experiments/Banknote3/sampler_cache_banknote.pkl"
_cached_data = {
    "samples": {},
    "model": None
}
_is_cache_loaded = False
_feature_names = ['variance', 'skewness', 'curtosis']
_banknote_data_3_features = None # Will hold the raw feature data

def _load_or_initialize_cache():
    """
    Loads data from the cache file if it exists. Otherwise, it generates
    the initial data, trains a model, and creates the cache file.
    """
    global _cached_data, _is_cache_loaded, _banknote_data_3_features

    if os.path.exists(_CACHE_FILENAME):
        print(f"--- Found cache file '{_CACHE_FILENAME}'. Loading data. ---")
        with open(_CACHE_FILENAME, 'rb') as f:
            _cached_data = pickle.load(f)
        # We still need to load the raw data for min/max calculations
        df = pd.read_csv(_DATA_FILE_PATH, header=None)
        df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
        _banknote_data_3_features = df[_feature_names]
            
    else:
        print(f"--- No cache file found. Initializing model and real samples. ---")
        try:
            df = pd.read_csv(_DATA_FILE_PATH, header=None)
        except FileNotFoundError:
            print(f"\nFATAL ERROR: Data file not found at '{_DATA_FILE_PATH}'. Please ensure the path is correct.\n")
            exit()

        df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
        
        _banknote_data_3_features = df[_feature_names]
        labels = df['class']
        
        model = DecisionTreeClassifier()
        # **FIX**: Convert DataFrame to NumPy array with .values to train without feature names
        model.fit(_banknote_data_3_features.values, labels)
        _cached_data["model"] = model

        label_name = 'class'
        for index, row in _banknote_data_3_features.iterrows():
            feature_values = row.values
            label_value = labels[index]
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

        min_vals = _banknote_data_3_features.min().values
        max_vals = _banknote_data_3_features.max().values
        
        for _ in range(num_synthetic_needed):
            synth_features = [random.uniform(min_vals[i], max_vals[i]) for i in range(len(_feature_names))]
            synth_label = _cached_data["model"].predict([synth_features])[0]
            
            sample_key = tuple((name, val) for name, val in zip(_feature_names, synth_features))
            sample_value = [('class', int(synth_label))]
            
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