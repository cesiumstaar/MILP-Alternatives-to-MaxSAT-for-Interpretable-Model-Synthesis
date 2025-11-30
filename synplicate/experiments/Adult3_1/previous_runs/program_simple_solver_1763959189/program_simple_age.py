import sys
sys.path.insert(0, '/home/lenin31/Desktop/repos/F1ML/synplicate/experiments/Adult3_1/')
import feature_defs

def execute(inputs):
    features = feature_defs.retrieve_feature_defs()
    value = features['age'](inputs)
    if value == 1:
        return 'income_<=50K'
    if value == 2:
        return 'income_<=50K'
    if value == 0:
        return 'income_<=50K'
    if value == 3:
        return 'income_<=50K'
    return 'income_<=50K'
