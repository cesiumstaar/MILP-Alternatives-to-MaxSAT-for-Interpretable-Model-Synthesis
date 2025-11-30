import sys
sys.path.insert(0, '/home/lenin31/Desktop/repos/F1ML/synplicate/experiments/ICML/AutoTaxi/')
import feature_defs

def execute(inputs):
    features = feature_defs.retrieve_feature_defs()
    value = features['init_pos'](inputs)
    if value == 3:
        return 'alert_1.0'
    if value == 2:
        return 'alert_0.0'
    return 'alert_1.0'
