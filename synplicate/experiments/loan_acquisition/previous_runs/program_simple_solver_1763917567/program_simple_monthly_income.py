import sys
sys.path.insert(0, '/home/lenin31/Desktop/repos/F1ML/synplicate/experiments/loan_acquisition/')
import feature_defs

def execute(inputs):
    features = feature_defs.retrieve_feature_defs()
    value = features['monthly_income'](inputs)
    if value == 0:
        return 'approved_0'
    if value == 1:
        return 'approved_1'
    return 'approved_0'
