import sys
sys.path.insert(0, '/home/lenin31/Desktop/repos/F1ML/synplicate/experiments/Iris3/')
import feature_defs

def execute(inputs):
    features = feature_defs.retrieve_feature_defs()
    value = features['sepal_length'](inputs)
    if value == 0:
        return 'species_1'
    if value == 1:
        return 'species_2'
    if value == 2:
        return 'species_2'
    if value == 3:
        return 'species_2'
    return 'species_2'
