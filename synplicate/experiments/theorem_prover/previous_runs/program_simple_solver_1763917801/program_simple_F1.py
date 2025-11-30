import sys
sys.path.insert(0, '/home/lenin31/Desktop/repos/F1ML/synplicate/experiments/theorem_prover/')
import feature_defs

def execute(inputs):
    features = feature_defs.retrieve_feature_defs()
    value = features['F1'](inputs)
    if value == 2:
        return 'H1_1'
    if value == 1:
        return 'H1_1'
    if value == 3:
        return 'H1_1'
    if value == 0:
        return 'H1_1'
    return 'H1_1'
