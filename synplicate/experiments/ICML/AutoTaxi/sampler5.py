from experiments.ICML.AutoTaxi import model as bb

import random
from random import choice
import math
random.seed(40)

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper



def uniform(num_of_samples):

    samples = {}

    # create 100 (features, Label) samples
    feature1_name = 'clouds'
    feature2_name = 'day_time'
    feature3_name = 'init_pos'
    feature4_name = 'init_heading'


    # print("----------------------")
    for i in range(num_of_samples):
        
        input_map = {}
        feature1_value = random.choice([random.randint(0,5)])
        # print(feature1_name, ": ", feature1_value,end =" | ")
        input_map[feature1_name] = feature1_value

        feature2_value = truncate(abs(random.uniform(50000.0,95000.0)),4)
        # print(feature2_name, ": ", feature2_value,end =" | ")
        input_map[feature2_name] = feature2_value

        feature3_value = truncate(abs(random.uniform(-8.0,8.0)),4)
        # print(feature2_name, ": ", feature2_value,end =" | ")
        input_map[feature3_name] = feature3_value

        # feature2_value = truncate(abs(random.uniform(50000.0,95000.0)),4)
        # # print(feature2_name, ": ", feature2_value,end =" | ")
        # input_map[feature2_name] = feature2_value
        
        
        prediction_value = bb.execute(input_map)
        # print(prediction_value)
        # print("----------------------")
        
        samples[(feature1_name,feature1_value),(feature2_name,feature2_value),(feature3_name,feature3_value)] = [("alert",prediction_value)]


    return samples