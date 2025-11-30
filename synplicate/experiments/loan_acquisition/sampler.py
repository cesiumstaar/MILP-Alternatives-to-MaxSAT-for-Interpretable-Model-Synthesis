# sampler.py for loan_acquisition with On-Disk Caching and Batch Prediction

import os
import pickle
import random
import math
import numpy as np
import pandas as pd
import tensorflow as tf

# --- Module-level Configuration ---
_MODEL_PATH = "experiments/loan_acquisition/model"
_CACHE_FILENAME = "experiments/loan_acquisition/sampler_cache_loan.pkl"

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
        
        feature1_name = 'age'
        feature2_name = 'monthly_income'
        feature3_name = 'dependents'
        feature4_name = 'credit_score'

        # --- Batch Generation (More Efficient) ---
        # 1. Create lists to hold all the raw data to be predicted
        ages_raw, incomes_raw, dependents_raw, scores_raw = [], [], [], []

        for _ in range(num_synthetic_needed):
            ages_raw.append(random.randint(18, 80))
            incomes_raw.append(truncate(abs(random.uniform(1000.0, 10000.0)), 2))
            dependents_raw.append(random.randint(0, 6))
            scores_raw.append(random.randint(300, 900))

        # 2. Make a single batch prediction
        print(f"--- Predicting {num_synthetic_needed} samples in a single batch... ---")
        predictions = model.predict({
            feature1_name: np.array(ages_raw),
            feature2_name: np.array(incomes_raw),
            feature3_name: np.array(dependents_raw),
            feature4_name: np.array(scores_raw)
        })
        print("--- Batch prediction complete. ---")

        # 3. Process the results and add to the cache
        for i in range(num_synthetic_needed):
            # Your logic to determine the class from the prediction output
            prediction_value = 1 if predictions[i][1] > predictions[i][0] else 0
            
            sample_key = (
                (feature1_name, ages_raw[i]),
                (feature2_name, incomes_raw[i]),
                (feature3_name, dependents_raw[i]),
                (feature4_name, scores_raw[i])
            )
            
            # Ensure the generated key is unique before adding
            while sample_key in _cached_samples:
                age_val = random.randint(18, 80)
                inc_val = truncate(abs(random.uniform(1000.0, 10000.0)), 2)
                dep_val = random.randint(0, 6)
                scr_val = random.randint(300, 900)
                sample_key = ((feature1_name, age_val), (feature2_name, inc_val), (feature3_name, dep_val), (feature4_name, scr_val))

            _cached_samples[sample_key] = [("approved", prediction_value)]

        # Save the updated cache back to the file
        with open(_CACHE_FILENAME, 'wb') as f:
            pickle.dump(_cached_samples, f)
        print(f"--- Cache updated. Saved {len(_cached_samples)} total samples to '{_CACHE_FILENAME}'. ---")

    all_samples_list = list(_cached_samples.items())
    random.shuffle(all_samples_list)
    
    return dict(all_samples_list[:num_of_samples])