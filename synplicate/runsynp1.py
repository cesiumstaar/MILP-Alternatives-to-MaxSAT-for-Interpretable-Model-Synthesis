import os
import importlib
import sys
import math
sys.path.append('../../')
sys.path.append('../')
from subprocess import STDOUT, PIPE, run
from distutils.util import strtobool


#import procedures
#from methods.erm import erm_procedure as erm
#from methods.cegqs import cegqs_procedure as cegqs

# import synthesizers
#from synthesizer.max_sharp_sat import mmc_synthesizer
from synthesizer.max_sat import ms_synthesizer
#from synthesizer.omt import omt_synthesizer
#from synthesizer.decision_trees import cart_tree_synthesizer


#from evaluator.recall_evaluator import evaluator
from evaluator.cochran_evaluator import evaluator as cochran

import timeit

from utils import logger

def writeConfig(dd_size, benchmark_path, features, fcount):
    cf = open (f"{benchmark_path}/config.mmc", 'w')
    cf.write (f"size =  {dd_size}\n")
    cf.write ("features = ") 
    index = 0;
    sz = len(features)
    for i, idx in zip(features, fcount):
        if idx > 0:
            nb = idx + 1
            cf.write(f"{i}:{nb}")
        else:
            cf.write(f"{i}:{idx}")
            
        #print (index, sz -1)
        if index < (sz - 1) :
            cf.write(", ")
        index+=1
    cf.write("\n")
    cf.write("labels = approved:2\n")
    cf.write("extend = false \n")
    cf.write ("feature_defs = feature_defs\n")
    cf.close()

def writeFeatureDefs(benchmark_path, features, fcount, labels, user_recommendation):
    sz = len(features)
    fd = open (f"{benchmark_path}/feature_defs.py", 'w')
    #fd = open (f"feature_defs.py", 'w')
    fd.write ("#Generated file \n")
    fd.write ("def check_num_of_inputs(inputs):\n")
    fd.write (f"\treturn len(inputs)=={sz}\n\n")
#write individual functions
    cindex = 0 
    for i in features:
        j = int(fcount[cindex])
        #print (i, j)
        #if j > 0:
        fd.write(f"def {i}(inputs):\n")
        fd.write ("\tassert(check_num_of_inputs(inputs))\n")
        fd.write (f"\tvalue = inputs[{cindex}][1]\n")
        findex = 0
        values = user_recommendation[cindex]
        values.sort(reverse=True)
        for val in values:
            fd.write (f"\tif value > {val}:\n")
            fd.write (f"\t\treturn {findex}\n")
            findex +=1
        fd.write(f"\treturn {findex}\n\n")
        cindex +=1
                  
    #currently assuming single label - have to change it to multiclass.
    lsz = len(labels)
    index = 0
    for i in labels:
        fd.write(f"def {i}(outputs):\n")
        fd.write (f"\tvalue = outputs[{index}][1]\n")
        fd.write (f"\treturn value\n\n")

    fd.write("def retrieve_feature_defs(): \n")
    fd.write("\tfeature_defs = {} \n")
    for j, c in zip(features, fcount):
        if c > 0:
            fd.write(f"\tfeature_defs[\"{j}\"] = {j}\n")
    for j in labels:
        fd.write(f"\tfeature_defs[\"{j}\"] = {j}\n")
    fd.write ( "\treturn feature_defs \n")
    fd.close()

def binarize(benchmark_path, features, user_choice, user_count, labels, useallx):
    sz = len(features)
    fd = open (f"{benchmark_path}/binarize.py", 'w')
    fd.write ("#Generated file to convert a numerical data file to binary \n")
#write individual functions
    cindex = 0 
    for i in features:
        j = int(user_count[cindex])
        #print (i, j)
        #if j > 0:
        fd.write(f"def {i}(inputs):\n")
        fd.write (f"\tvalue = inputs[{cindex}][1]\n")
        findex = 0
        values = user_choice[cindex]
        values.sort(reverse=True)
        numv = len(values) + 1
        for val in values:
            fd.write (f"\tif value > {val}:\n")
            fd.write (f"\t\treturn ")
            for b in range(numv):
                if b == findex:
                    fd.write(" 1")
                else:
                    fd.write(" 0")
                if b < numv - 1:
                    fd.write(",")
                else:
                    fd.write("\n")
            findex +=1
        fd.write(f"\treturn ")
        for b in range(numv):
            if b == findex:
                fd.write(" 1")
            else:
                fd.write(" 0")
            if b < numv - 1:
                fd.write(",")
            else:
                fd.write("\n\n")
        cindex +=1
                  
    #currently assuming single label - have to change it to multiclass.
    lsz = len(labels)
    if lsz > 2:
        print ("dl8.5 works only on binary classification problems!! ")
        return False
    index = 0
    for i in labels:
        fd.write(f"def {i}(outputs):\n")
        fd.write (f"\tvalue = outputs[{index}][1]\n")
        fd.write (f"\treturn value\n\n")

#Write fix_nulls

    fd.write("def fix_nulls(s):\n")
    fd.write("\tfor line in s:\n")
    fd.write(" \t\tyield line.replace(\'\\0\', \"\")\n\n")
    
    fd.write("def convert(file_name):\n")
    fd.write("\toFile_name =  os.path.splitext(file_name)[0]+\'.dl85.txt\'\n")
    fd.write("\twith open(file_name, \'r\') as i_file, open(oFile_name, \'w\') as  o_file:\n")
    fd.write("\t\tcsv_reader = csv.reader(fix_nulls(i_file), delimiter=\',\')\n")
    fd.write("\t\tcsv_writer = csv.writer(o_file, delimiter=\' \')\n")
    fd.write("\t\tline_count = 0 \n")
    fd.write("\t\tfor row in csv_reader():\n")
    fd.write("\t\t\tif line_count == 0: \n")
    fd.write("\t\t\t\tline_count += 1\n")
    fd.write("\t\t\telse:\n")
    findex = 0
    cindex = 0
    for f in features:
            fd.write("\t\t\t\t")
            for j in range(int(user_count[findex])):
                fd.write(f"d{cindex},")
                cindex +=1
            fd.write(f"d{cindex} = {f}(float(row[{findex}]))\n")
            cindex +=1
            findex += 1
    fd.write(f"\t\t\t\td{cindex} = {labels[0]}(int(row[{findex}]))\n") #Since DL8.5 supports only one output
    xlimit = cindex - 1
    fd.write("\t\t\t\tnew_row = [")
    #cindex += 1
    print (cindex)
    for ci in range(cindex):
        fd.write(f"d{ci}, ")
    fd.write(f"d{cindex}]\n")
    fd.write("\t\t\t\tcsv_writer.writerow(new_row)\n")
    fd.write("\t\t\t\tline_count += 1\n")
    fd.write("\t\to_file.close()\n")
    fd.write("\t\ti_file.close()\n")
    
    print(xlimit)
    fd.close()



def query_yes_no(question, default='no'):
    if default is None:
        prompt = " [y/n] "
    elif default == 'yes':
        prompt = " [Y/n] "
    elif default == 'no':
        prompt = " [y/N] "
    else:
        raise ValueError(f"Unknown setting '{default}' for default.")

    while True:
        try:
            resp = input(question + prompt).strip().lower()
            if default is not None and resp == '':
                return default == 'yes'
            else:
                return strtobool(resp)
        except ValueError:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")

def genTrainingData(class_size, delta, epsilon, synthesis_benchmark_path):
#    class_size = synthesizer.compute_class_size() #Not possible to do this anymore
    num_of_samples = math.ceil(math.log((1+class_size)/delta)/epsilon)
    print(f"Using {num_of_samples} samples ...")
    sampler = importlib.import_module(f".sampler",synthesis_benchmark_path.replace("/",".").rstrip('.'))
    samples = {}
    #num_of_samples = 100    # for now
    samples.update(sampler.uniform(num_of_samples))
    logger.dump_samples(samples,synthesis_benchmark_path,f"erm_syn_samples")
    return samples

def genTestData(delta, epsilon, benchmark_path):

    z_value = 0
    if delta == 0.05:
        z_value = 1.96
    elif delta == 0.01:
        z_value = 2.33
    else:
        raise ValueError(f"z-value for delta {delta} unknown")

    sampler = importlib.import_module(f".sampler",benchmark_path.replace("/",".").rstrip('.'))
    num_of_samples = math.ceil((z_value**2)*0.25/(epsilon**2))
    #num_of_samples = 100# for now
    print(f"|---Using {num_of_samples} samples for test data")
    eva_samples = {}
    # Retrieve samples 
    eva_samples.update(sampler.uniform(num_of_samples))
    logger.dump_samples(eva_samples,benchmark_path,f"dt_samples")
    return eva_samples


def experiment(dd_size, delta, epsilon, dir_name,depth, class_size, features, labels):

    synthesis_benchmark_path = f"experiments/loan_acquisition/"
    print (synthesis_benchmark_path)
    test = synthesis_benchmark_path.replace("/",".").rstrip('.')
    print (test)

    erm_synthesis_time = 0

# Generate training and test data
    print ("Generating Training Data ... " )
    samples = genTrainingData(class_size, delta, epsilon, synthesis_benchmark_path)
    print ("Generating Test Data ... " )
    #test_data = genTestData(delta, epsilon, synthesis_benchmark_path )

    
    samples_file = f"{synthesis_benchmark_path}samples/erm_syn_samples.csv"
    test_file = f"{synthesis_benchmark_path}samples/dt_samples.csv"

#Run a opt decision tree tool 
    print ("Running Infer DT..............Please wait")
    #result = run (f"./synthesizer/InferDT/InferDT {samples_file} -v -x infer -t {test_file}", stdout=PIPE, stderr=STDOUT, universal_newlines=True, shell=True, timeout=300) #Timeout for 5 minutes
    result = run (f"./synthesizer/InferDT/InferDT {samples_file} -v -x infer", stdout=PIPE, stderr=STDOUT, universal_newlines=True, shell=True)
    print ("Result of Infer DT")
    res = result.stdout
    print (res)
    # printing result 
    strres = res.split()
    user_recommendation=[]
    fcount = []

    recommend = False;
    for f in features:
        count = 0
        for sentence in strres:
            if sentence.count(f) > 0:
                if not sentence.startswith("!"):
                    sent = sentence.replace(":", "")
                    user_recommendation.append(sent)
                    count +=1
        fcount.append (count)
        if count > 0:       #The output contains the tree
            recommend = True;
#Take user choices and recommendations
    print ("Recommended Features and Partitions Using InferDT")
    print (user_recommendation)
    print (fcount)
    findex = 0
    rindex = 0
    user_choice = [] 
    user_count = []
    for i in fcount:
        reco = user_recommendation[rindex:rindex+i]
        values = []
        for e in reco:
            tt = e.split(">")
            values.append(tt[1])
        values.sort(reverse=True)
        print (f"For feature {features[findex]}, recommended values are" )
        print(values)
        x =  input ("Please give the values of your interest. If you want to use the recommended values, press enter.): ").split()
        if (len(x) == 0):
            user_choice.append(values)
            user_count.append (len(values))
        else:
            user_choice.append(x)
            user_count.append(len(x))
        findex +=1
        rindex +=i

    print ("Your choices are " )
    print (user_choice)
    print (user_count)

#Generate feature_defs.py
    writeFeatureDefs(synthesis_benchmark_path, features, user_count, labels, user_choice)
    binarize(synthesis_benchmark_path, features, user_choice, user_count, labels, True)

    for l in range (dd_size, 1, -1):
#create config.mmc
        writeConfig(l, synthesis_benchmark_path, features, fcount)
    #--------------Calling the synthesizer ------#
        synthesis_time = 0

        synthesizer.initialize(synthesis_benchmark_path)

        start = timeit.default_timer()
        print (f"Synthesizing program of size {l}")
        erm_program_path, dot_path, count = synthesizer.synthesize(synthesis_benchmark_path,samples,f"ms_erm")
        if erm_program_path != "" :
            print(f"Synthesized program: {erm_program_path}\nVisualization: {dot_path}")
            stop = timeit.default_timer()
            synthesis_time = stop-start
            num_of_samples = len(samples)
            print(f"Synthesis time:{synthesis_time}")
            if count>0:
                print(f"Misclassification rate:{1-count/num_of_samples}")

            print("----------------------------")
            print("Evaluating program on test dataset...")
                       
           #refinement, misclass_rate  = cochran.eval(test_data,synthesis_benchmark_path,erm_program_path, samples, delta, epsilon, 100)
            #print("****************************************************")
            #print(f"ERM misclass rate: {misclass_rate}. Time: {erm_synthesis_time}. Size: {depth}")
            #print("****************************************************")



dd_size = 1
delta = 0.05
epsilon = 0.05
benchmark_name = "la"
class_size=100
synthesizer = ms_synthesizer
dir_name = f"{benchmark_name}_{dd_size}_{delta}_{epsilon}"
features= [ "age", "monthly_income","dependents", "credit_score"]
labels =["approved"]

experiment(dd_size,delta,epsilon,dir_name,dd_size, class_size, features, labels)
