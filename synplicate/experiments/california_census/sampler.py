# sampler.py for california_census with On-Disk Caching and Batch Prediction (Corrected)

import os
import pickle
import random
import math
import numpy as np
import pandas as pd
import tensorflow as tf

# --- Module-level Configuration ---
_MODEL_PATH = "experiments/california_census/model"
_DATA_FILE_PATH = "experiments/california_census/california_housing_train_classifier.csv"
_CACHE_FILENAME = "experiments/california_census/sampler_cache_tf.pkl"

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

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def uniform(num_of_samples):
    """
    Provides samples from the on-disk cache. If more samples are requested
    than exist, it generates new synthetic ones using batch prediction and updates the cache file.
    """
    global _cached_samples

    if not _is_cache_loaded:
        _load_or_initialize_cache()

    if num_of_samples > len(_cached_samples):
        num_synthetic_needed = num_of_samples - len(_cached_samples)
        print(f"\n--- Cache has {len(_cached_samples)} samples. Requested {num_of_samples}. Generating {num_synthetic_needed} new synthetic samples. ---\n")

        # Load the pre-trained TensorFlow model
        model = tf.keras.models.load_model(_MODEL_PATH)
        
        # Get normalization stats from the original training data
        train_df = pd.read_csv(_DATA_FILE_PATH)
        train_df_mean = train_df.mean()
        train_df_std = train_df.std()

        feature3_name = 'population'
        feature4_name = 'median_income'

        # --- Batch Generation (More Efficient) ---
        populations_raw = []
        incomes_raw = []
        populations_norm = []
        incomes_norm = []

        for _ in range(num_synthetic_needed):
            pop_val = random.randint(3, 35682)
            inc_val = random.uniform(0.5, 15.0)
            
            pop_norm = truncate((pop_val - train_df_mean[feature3_name]) / train_df_std[feature3_name], 2)
            inc_norm = truncate((inc_val - train_df_mean[feature4_name]) / train_df_std[feature4_name], 2)
            
            populations_raw.append(pop_val)
            incomes_raw.append(inc_val)
            populations_norm.append(pop_norm)
            incomes_norm.append(inc_norm)

        print(f"--- Predicting {num_synthetic_needed} samples in a single batch... ---")
        predictions = model.predict({
            feature3_name: np.array(populations_norm),
            feature4_name: np.array(incomes_norm)
        })
        print("--- Batch prediction complete. ---")

        for i in range(num_synthetic_needed):
            predicted_class = np.argmax(predictions[i])
            sample_key = ((feature3_name, populations_raw[i]), (feature4_name, incomes_raw[i]))
            
            while sample_key in _cached_samples:
                 pop_val = random.randint(3, 35682)
                 inc_val = random.uniform(0.5, 15.0)
                 sample_key = ((feature3_name, pop_val), (feature4_name, inc_val))

            _cached_samples[sample_key] = [("Class", predicted_class)]

        # **FIX**: Use the correct variable name '_cached_samples'
        with open(_CACHE_FILENAME, 'wb') as f:
            pickle.dump(_cached_samples, f)
        print(f"--- Cache updated. Saved {len(_cached_samples)} total samples to '{_CACHE_FILENAME}'. ---")

    all_samples_list = list(_cached_samples.items())
    random.shuffle(all_samples_list)
    
    return dict(all_samples_list[:num_of_samples])