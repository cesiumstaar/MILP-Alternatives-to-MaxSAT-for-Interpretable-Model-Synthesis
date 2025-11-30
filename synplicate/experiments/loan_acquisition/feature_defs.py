# inputs: [longitude, latitude, population, median_income]
# features: population, median_income, location
# output: [median_house_price]
# labels: median_house_price

def check_num_of_inputs(inputs):
    return len(inputs)==4

def age(inputs):
    assert(check_num_of_inputs(inputs))
    value = inputs[0][1]
    if value<30:
        return 0
    elif value < 50:
        return 1
    elif value <60:
        return 2
    else:
        return 3

def monthly_income(inputs):
    assert(check_num_of_inputs(inputs))
    value = inputs[1][1] 
    if value<6000.0:
        return 0
    else:
        return 1

def dependents(inputs):
    assert(check_num_of_inputs(inputs))
    value = inputs[2][1] 
    if value<3:
        return 0
    else:
        return 1

def credit_score(inputs):
    assert(check_num_of_inputs(inputs))
    value = inputs[3][1] 
    if value<500:
        return 0
    elif value<700:
        return 1
    else:
        return 2


def approved(outputs):
    value = outputs[0][1]   
    return value


def retrieve_feature_defs():
    feature_defs = {}
    feature_defs["age"] = age
    feature_defs["monthly_income"] = monthly_income
    feature_defs["dependents"] = dependents
    feature_defs["credit_score"] = credit_score
    feature_defs["approved"] = approved

    return feature_defs