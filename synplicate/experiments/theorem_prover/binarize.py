import csv
import sys
import os

def F1(value):
    if value< 0.25:
        return 0, 0, 0, 1
    elif value < 0.50:
        return 0, 0, 1, 0
    elif value < 0.75:
        return 0, 1, 0, 0
    else:
        return 1, 0, 0, 0


def F10(value):
    if value<2.0:
        return 0,0, 1
    elif value <3.0:
        return 0,1,0
    else:
        return 1,0,0

def H1(value):
    return value

def retrieve_feature_defs():
    feature_defs = {}
    feature_defs["F1"] = F1
    feature_defs["F10"] = F10
    feature_defs["H1"] = H1

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
#            csv_writer.writerow(["F1_1", "F1_2", "F10_1", "F10_2", row[2]])
#            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            bin1, bin2, bin3, bin4 = F1(float(row[0]))
            bin5, bin6, bin7 = F10(float(row[1]))
            bin8 = H1(float(row[2]))
            new_row = [bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8]
            csv_writer.writerow(new_row)
            line_count += 1
    print(f'Processed {line_count} lines.')
