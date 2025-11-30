import csv
import sys
import os

def age(value):
    if value<30:
        return 0, 0, 0, 1
    elif value < 50:
        return 0, 0, 1, 0
    elif value <60:
        return 0, 1, 0, 0
    else:
        return  1, 0, 0, 0

def monthly_income(value):
    if value<6000.0:
        return 0
    else:
        return 1

def dependents(value):
    if value<3:
        return 0
    else:
        return 1

def credit_score(value):
    if value<500:
        return 0, 0, 1
    elif value<700:
        return 0, 1, 0
    else:
        return 1, 0, 0


def approved(value):
    return value

def retrieve_feature_defs():
    feature_defs = {}
    feature_defs["age"] = age
    feature_defs["monthly_income"] = monthly_income
    feature_defs["dependents"] = dependents
    feature_defs["credit_score"] = credit_score
    feature_defs["approved"] = approved

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
            #csv_writer.writerow(["Age_1", "Age_2", "Monthly_Income", row[2], "Credit_Score_1", "Credit_Score_2", row[4]])
#            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            bin1, bin2, bin3, bin4 = age(float(row[0]))
            bin5 = monthly_income(float(row[1]))
            bin6 = dependents(float(row[2]))
            bin7, bin8, bin9 = credit_score(int(row[3]))
            bin10 = approved(int(row[4]))
            new_row = [bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8, bin9, bin10]
            csv_writer.writerow(new_row)
            line_count += 1
    print(f'Processed {line_count} lines.')
