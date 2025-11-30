# inputs: [longitude, latitude, population, median_income]
# features: population, median_income, location
# output: [median_house_price]
# labels: median_house_price

def check_num_of_inputs(inputs):
    return len(inputs)==2

def population(inputs):
    assert(check_num_of_inputs(inputs))
    value = inputs[0][1]
    if value<3000.0:
        return 1
    if value<10000.0:
        return 2
    if value<20000.0:
        return 3
    else:
        return 0

def median_income(inputs):
    assert(check_num_of_inputs(inputs))
    value = inputs[1][1] 
    if value<7.0:
        return 1
    else:
        return 0

# def location(inputs):
#     assert(check_num_of_inputs(inputs))
#     longitude = inputs[0][1]
#     latitude = inputs[1][1]
#     if -114.0>=longitude and longitude>-118.0:
#         if 36.0<=latitude and latitude<39:
#             return 1
#     if -118.0>= longitude and longitude>=-124.3:
#         if 39<=latitude and latitude<=42.5:
#             return 2    
#     return 0

def Class(outputs):
    value = outputs[0][1]   
    return value

def test(outputs):
    return 0

def retrieve_feature_defs():
    feature_defs = {}
    feature_defs["population"] = population
    feature_defs["median_income"] = median_income
    # feature_defs["location"] = location
    feature_defs["Class"] = Class
    # feature_defs["test"] = test

    return feature_defs