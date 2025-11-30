# inputs: [longitude, latitude, population, median_income]
# features: population, median_income, location
# output: [median_house_price]
# labels: median_house_price

import csv
import sys
import os

def population(value):
    if value<3000.0:
        return 0, 0, 0, 1
    if value<10000.0:
        return 0, 0, 1, 0 
    if value<20000.0:
        return 0, 1, 0, 0
    else:
        return 1, 0, 0, 0

def median_income(value):
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

def class_label(value):
    if value < 2:
        return 1
    return 0

def retrieve_feature_defs():
    feature_defs = {}
    feature_defs["population"] = population
    feature_defs["median_income"] = median_income
    # feature_defs["location"] = location
    feature_defs["Class"] = Class
    # feature_defs["test"] = test

    return feature_defs

def fix_nulls(s):
    for line in s:
        yield line.replace('\0', "")

file_name = sys.argv[1]
oFile_name =  os.path.splitext(file_name)[0]+'.dl85.txt'
with open(file_name, 'r') as i_file, open(oFile_name, 'w') as o_file:
    csv_reader = csv.reader(fix_nulls(i_file), delimiter=',')
    csv_writer = csv.writer(o_file, delimiter=' ')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
#            csv_writer.writerow(["Feat1_1" , "Feat1_2", row[1], "Class_1", "Class_2"])
#            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            bin1, bin2, bin3, bin4 = population(float(row[0]))
            bin5 = median_income(float(row[1]))
            bin6  = class_label(int(row[2]))
            new_row = [bin1, bin2, bin3, bin4, bin5, bin6]
            csv_writer.writerow(new_row)
            line_count += 1
    print(f'Processed {line_count} lines.')
