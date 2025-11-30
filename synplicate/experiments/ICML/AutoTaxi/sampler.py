import pandas as pd
import random

def uniform(num_of_samples):
    """
    Directly reads the data.csv file, formats it, and returns a
    random subset of the requested size on every run.
    """
    
    csv_source_file = "experiments/ICML/AutoTaxi/data.csv"

    try:
        df = pd.read_csv(csv_source_file)
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Source CSV file not found at '{csv_source_file}'.\n")
        return {}

    # **FIX**: Convert specific columns to integers to ensure correct formatting.
    df['clouds'] = df['clouds'].astype(int)
    df['alert'] = df['alert'].astype(int)

    feature_names = ['clouds', 'day_time', 'init_pos']
    label_name = 'alert'
    
    samples = {}
    for index, row in df.iterrows():
        sample_key = tuple(
            (name, row[name]) for name in feature_names
        )
        
        label_value = row[label_name]
        sample_value = [(label_name, label_value)]
        
        samples[sample_key] = sample_value
    
    if num_of_samples >= len(samples):
        return samples

    all_samples_list = list(samples.items())
    random.shuffle(all_samples_list)
    
    return dict(all_samples_list[:num_of_samples])