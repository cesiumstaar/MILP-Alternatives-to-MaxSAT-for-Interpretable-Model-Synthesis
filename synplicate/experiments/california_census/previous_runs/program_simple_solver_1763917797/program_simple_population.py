import sys
sys.path.insert(0, '/home/lenin31/Desktop/repos/F1ML/synplicate/experiments/california_census/')
import feature_defs

def execute(inputs):
    features = feature_defs.retrieve_feature_defs()
    value = features['population'](inputs)
    if value == 0:
        return 'Class_1'
    if value == 1:
        return 'Class_3'
    if value == 2:
        return 'Class_1'
    if value == 3:
        return 'Class_1'
    return 'Class_1'
