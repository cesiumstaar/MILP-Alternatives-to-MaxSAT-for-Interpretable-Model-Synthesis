# sampler.py for the 3-feature Adult dataset (On-Disk Caching Version - Corrected)

import os
import pickle
import random
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder

# --- Module-level Configuration ---
_DATA_FILE_PATH = "experiments/Adult3_1/adult.data"
_CACHE_FILENAME = "experiments/Adult3_1/sampler_cache_adult_3_features.pkl"
_cached_data = {
    "samples": {},
    "model": None,
    "encoder": None
}
_is_cache_loaded = False
_feature_names = ['age', 'hours_per_week', 'workclass']
_numeric_features = ['age', 'hours_per_week']
_categorical_features = ['workclass']
_raw_data = None

def _load_or_initialize_cache():
    """
    Loads data from the cache file if it exists. Otherwise, it generates
    the initial data, trains a model, and creates the cache file.
    """
    global _cached_data, _is_cache_loaded, _raw_data

    if os.path.exists(_CACHE_FILENAME):
        print(f"--- Found cache file '{_CACHE_FILENAME}'. Loading data. ---")
        with open(_CACHE_FILENAME, 'rb') as f:
            _cached_data = pickle.load(f)
        col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
                     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
                     'hours-per-week', 'native-country', 'income']
        _raw_data = pd.read_csv(_DATA_FILE_PATH, header=None, names=col_names, sep=',\s*', engine='python', na_values='?')
        _raw_data.rename(columns={'hours-per-week': 'hours_per_week'}, inplace=True)
        _raw_data.dropna(inplace=True)

    else:
        print(f"--- No cache file found. Initializing model and real samples. ---")
        try:
            col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
                         'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
                         'hours-per-week', 'native-country', 'income']
            df = pd.read_csv(_DATA_FILE_PATH, header=None, names=col_names, sep=',\s*', engine='python', na_values='?')
        except FileNotFoundError:
            print(f"\nFATAL ERROR: Data file not found at '{_DATA_FILE_PATH}'. Please ensure path is correct.\n")
            exit()
        
        df.rename(columns={'hours-per-week': 'hours_per_week'}, inplace=True)
        df.dropna(inplace=True)

        # **FIX**: Group 'workclass' categories before any other processing
        gov_group = ['Local-gov', 'State-gov', 'Federal-gov']
        unemployed_group = ['Without-pay', 'Never-worked']
        df['workclass'] = df['workclass'].replace(gov_group, 'Government')
        df['workclass'] = df['workclass'].replace(unemployed_group, 'Unemployed')

        df_subset = df[_feature_names + ['income']]
        _raw_data = df_subset
        
        X = df_subset[_feature_names]
        y = df_subset['income'].map({'<=50K': 0, '>50K': 1})
        
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_encoded_cat = encoder.fit_transform(X[_categorical_features].values)
        _cached_data["encoder"] = encoder
        
        X_processed = np.concatenate([X[_numeric_features].values, X_encoded_cat], axis=1)

        model = DecisionTreeClassifier()
        model.fit(X_processed, y)
        _cached_data["model"] = model

        for index, row in X.iterrows():
            feature_values = row.values
            label_value = df_subset.loc[index, 'income']
            sample_key = tuple((name, val) for name, val in zip(_feature_names, feature_values))
            sample_value = [('income', label_value)]
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

        min_vals_num = _raw_data[_numeric_features].min().values
        max_vals_num = _raw_data[_numeric_features].max().values
        
        categories = _cached_data["encoder"].categories_
        label_map_rev = {0: '<=50K', 1: '>50K'}

        for _ in range(num_synthetic_needed):
            synth_num = [random.uniform(min_vals_num[i], max_vals_num[i]) for i in range(len(_numeric_features))]
            synth_cat_text = [random.choice(cat) for cat in categories]
            
            synth_cat_encoded = _cached_data["encoder"].transform([synth_cat_text])
            synth_processed = np.concatenate([np.array([synth_num]), synth_cat_encoded], axis=1)
            
            synth_label_encoded = _cached_data["model"].predict(synth_processed)[0]
            synth_label_text = label_map_rev[synth_label_encoded]
            
            synth_features_original_style = synth_num + synth_cat_text
            sample_key = tuple((name, val) for name, val in zip(_feature_names, synth_features_original_style))
            sample_value = [('income', synth_label_text)]
            
            while sample_key in _cached_data["samples"]:
                synth_num = [random.uniform(min_vals_num[i], max_vals_num[i]) for i in range(len(_numeric_features))]
                synth_cat_text = [random.choice(cat) for cat in categories]
                synth_features_original_style = synth_num + synth_cat_text
                sample_key = tuple((name, val) for name, val in zip(_feature_names, synth_features_original_style))

            _cached_data["samples"][sample_key] = sample_value

        with open(_CACHE_FILENAME, 'wb') as f:
            pickle.dump(_cached_data, f)
        print(f"--- Cache updated. Saved {len(_cached_data['samples'])} total samples to '{_CACHE_FILENAME}'. ---")

    all_samples_list = list(_cached_data["samples"].items())
    random.shuffle(all_samples_list)
    
    return dict(all_samples_list[:num_of_samples])