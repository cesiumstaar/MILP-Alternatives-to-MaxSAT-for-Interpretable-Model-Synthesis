# inputs: [longitude, latitude, population, median_income]
# features: population, median_income, location
# output: [median_house_price]
# labels: median_house_price

def check_num_of_inputs(inputs):
    return len(inputs)==2

def F1(inputs):
    assert(check_num_of_inputs(inputs))
    value = inputs[0][1]
    if value< 0.25:
        return 0
    elif value < 0.50:
        return 1
    elif value < 0.75:
        return 2
    else:
        return 3

def F10(inputs):
    assert(check_num_of_inputs(inputs))
    value = inputs[1][1] 
    if value<2.0:
        return 0
    elif value <3.0:
        return 1
    else:
        return 2


def H1(outputs):
    value = outputs[0][1]   
    return value


def retrieve_feature_defs():
    feature_defs = {}
    feature_defs["F1"] = F1
    feature_defs["F10"] = F10
    feature_defs["H1"] = H1

    return feature_defs