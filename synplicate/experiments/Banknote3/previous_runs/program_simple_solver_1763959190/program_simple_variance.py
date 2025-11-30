import sys
sys.path.insert(0, '/home/lenin31/Desktop/repos/F1ML/synplicate/experiments/Banknote3/')
import feature_defs

def execute(inputs):
    features = feature_defs.retrieve_feature_defs()
    value = features['variance'](inputs)
    if value == 3:
        return 'class_0'
    if value == 2:
        return 'class_0'
    if value == 1:
        return 'class_1'
    if value == 0:
        return 'class_1'
    return 'class_0'
