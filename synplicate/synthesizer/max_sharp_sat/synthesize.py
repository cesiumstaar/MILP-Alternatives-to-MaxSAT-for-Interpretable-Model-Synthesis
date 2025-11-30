import tensorflow as tf
import sys
import os
sys.path.append('../')
from sharpsat import encoder


path = "../examples/california_census_simplified/model"
model = tf.keras.models.load_model(path)

encoder.encode(model,3,4)

os.system("./bc2cnf -all encoding.sat encoding_temp.dimacs")

dimacs_file = open("encoding.dimacs","w")
dimacs_file.write("c max")
maxvars = encoder.extract_max_variables("encoding_temp.dimacs")
for s in maxvars:
    dimacs_file.write(f" {s} ")
dimacs_file.write("0\n")

dimacs_file.write("c ind")
indvars = encoder.extract_ind_variables("encoding_temp.dimacs")
for s in indvars:
    dimacs_file.write(f" {s} ")
dimacs_file.write("0\n")

dimacs_temp_file = open("encoding_temp.dimacs","r")
temp_string = dimacs_temp_file.read()

dimacs_file.write(temp_string)

dimacs_file.close()
dimacs_temp_file.close()

os.system("rm encoding_temp.dimacs")

os.system("python maxcount.py encoding.dimacs 1")

print("Done")