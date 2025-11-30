
import importlib
import os
from synthesizer.max_sharp_sat import encoder


def synthesize(benchmark_path,samples,iteration):
    
    # create max#sat encoding
    encoding_file_name = f"encoding{iteration}"
    output_path = benchmark_path+"maxsat_encoding/"
    # TODO create encoding 
    encoding_path = "" # TODO call encoder. use output path as directory to store encoding 

    # maxsat 
    print("Max Sat...")
    witness_path = output_path + f"witness{iteration}.txt"
    # TODO safe program into witness file 

    # translate witness to program: extract program from maxsat witness
    # extract program signature 
    input_names = []
    for s in samples.keys():
        for i in range(len(s)):
            input_names.append(s[i][0])
        break
    # print(input_names)
    program_path = ""
    dot_path = ""
    # TODO extract program from witness


    return program_path, dot_path