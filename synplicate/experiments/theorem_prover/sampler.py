# sampler.py for theorem_prover with On-Disk Caching (Corrected)

import os
import pickle
import random
import math
import numpy as np
import pandas as pd
import tensorflow as tf

# --- Module-level Configuration ---
_MODEL_PATH = "experiments/theorem_prover/model"
_CACHE_FILENAME = "experiments/theorem_prover/sampler_cache_theorem_prover.pkl"

_cached_samples = {}
_is_cache_loaded = False

def _load_or_initialize_cache():
    """
    Loads samples from the cache file if it exists. Otherwise, it creates
    an empty cache and saves it.
    """
    global _cached_samples, _is_cache_loaded
    if os.path.exists(_CACHE_FILENAME):
        print(f"--- Found cache file '{_CACHE_FILENAME}'. Loading samples. ---")
        with open(_CACHE_FILENAME, 'rb') as f:
            _cached_samples = pickle.load(f)
    else:
        print(f"--- No cache file found. Initializing empty cache. ---")
        _cached_samples = {}
        with open(_CACHE_FILENAME, 'wb') as f:
            pickle.dump(_cached_samples, f)
    _is_cache_loaded = True

def uniform(num_of_samples):
    """
    Provides samples from the on-disk cache. If more samples are requested
    than exist, it generates new synthetic ones until all possible combos are found.
    """
    global _cached_samples

    if not _is_cache_loaded:
        _load_or_initialize_cache()
        
    # **FIX**: Calculate the maximum possible number of unique samples
    max_f1_options = 100 - 10 + 1  # 91
    max_f10_options = 50 - 10 + 1 # 41
    max_possible_samples = max_f1_options * max_f10_options # 3731

    if num_of_samples > len(_cached_samples):
        # Check if we have already found all possible unique samples
        if len(_cached_samples) >= max_possible_samples:
            print(f"\n--- WARNING: All {max_possible_samples} unique feature combinations have been generated. Cannot generate more. ---")
            print(f"--- Returning all available samples. ---")
        else:
            num_synthetic_needed = num_of_samples - len(_cached_samples)
            print(f"\n--- Cache has {len(_cached_samples)} samples. Requested {num_of_samples}. Generating up to {num_synthetic_needed} new synthetic samples. ---\n")

            model = tf.keras.models.load_model(_MODEL_PATH)
            
            feature1_name = 'F1'
            feature2_name = 'F10'
            
            f1_raw, f10_raw = [], []

            for _ in range(num_synthetic_needed):
                # If we've hit the max, stop trying to generate more.
                if len(_cached_samples) + len(f1_raw) >= max_possible_samples:
                    print(f"\n--- NOTE: Reached max of {max_possible_samples} unique samples during generation. Stopping. ---")
                    break

                f1_val = random.randint(10, 100) / 100.0
                f10_val = random.randint(10, 50) / 10.0
                sample_key = ((feature1_name, f1_val), (feature2_name, f10_val))

                while sample_key in _cached_samples or sample_key in [( (f1_raw[i], f1_raw[i]),(f10_raw[i], f10_raw[i]) ) for i in range(len(f1_raw))]:
                    f1_val = random.randint(10, 100) / 100.0
                    f10_val = random.randint(10, 50) / 10.0
                    sample_key = ((feature1_name, f1_val), (feature2_name, f10_val))
                
                f1_raw.append(f1_val)
                f10_raw.append(f10_val)

            # Only run prediction if new samples were actually generated
            if f1_raw:
                print(f"--- Predicting {len(f1_raw)} new samples in a single batch... ---")
                predictions = model.predict({
                    feature1_name: np.array(f1_raw),
                    feature2_name: np.array(f10_raw)
                })
                print("--- Batch prediction complete. ---")

                for i in range(len(f1_raw)):
                    prediction_value = 0 if predictions[i][0] > predictions[i][1] else 1
                    sample_key = ((feature1_name, f1_raw[i]), (feature2_name, f10_raw[i]))
                    _cached_samples[sample_key] = [("H1", prediction_value)]

                with open(_CACHE_FILENAME, 'wb') as f:
                    pickle.dump(_cached_samples, f)
                print(f"--- Cache updated. Saved {len(_cached_samples)} total samples to '{_CACHE_FILENAME}'. ---")

    all_samples_list = list(_cached_samples.items())
    random.shuffle(all_samples_list)
    
    return dict(all_samples_list[:num_of_samples])