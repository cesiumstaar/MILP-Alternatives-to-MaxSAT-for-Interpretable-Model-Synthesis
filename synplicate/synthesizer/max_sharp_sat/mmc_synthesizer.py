
import importlib
import os
from synthesizer.max_sharp_sat import encoder
from subprocess import STDOUT, PIPE, run



# global variables 
num_of_feature_nodes = 0
feature_partition = {}
label_partition = {}
feature_defs = {}
output_path = ""

def configure(path):
    config_file = open(f"{path}config.mmc","r")
    # initialize 
    size = 0
    feature_partition = {}
    label_partition = {}
    extend = False
    feature_defs = {}
    # parse config file
    for line in config_file:
        word_list = line.split()
        if word_list[0]=="size":
            size = int(word_list[2])
        elif word_list[0]=="features":
            for feature in word_list[2:]:
                name = feature.split(':')[0] 
                num_of_partitions = feature.split(':')[1].split(',')[0] 
                feature_partition[name] = int(num_of_partitions)
        elif word_list[0]=="labels":
            for label in word_list[2:]:
                name = label.split(':')[0] 
                num_of_partitions = label.split(':')[1].split(',')[0] 
                label_partition[name] = int(num_of_partitions)
        elif word_list[0]=="extend":
            if word_list[2] == "True":
                extend = True
        elif word_list[0]=="feature_defs":
            module = importlib.import_module(f".{word_list[2]}",path.replace("/",".").rstrip('.'))
            feature_defs = module.retrieve_feature_defs()
        else:
            raise Exception("Config file syntax error!")
        # print(word_list[0])

    return (size,feature_partition,label_partition,feature_defs,extend)

def extract_program_map(encoding_path,witness_path):

    witness_file = open(witness_path,"r")
    var_ids = []
    count = 0
    for line in witness_file.readlines():
        if line.startswith("v "):
            # extract variable id 
            current_var_id = ""
            for letter in line:
                if letter == '\n':
                    break
                elif letter.isdigit() or letter=='-':
                    current_var_id += letter
                else:
                    if current_var_id!="":
                         var_ids.append(current_var_id)
                    current_var_id = ""
            print("Witness found! ",end='')
        if line.startswith("c Estimated max-count"):
            words = line.split()
            count = int(float(words[3])* (float(words[5].split('^')[0])**float(words[5].split('^')[1])))
            print("Count: ", count)
        
    if var_ids == []:
        print("No witness found!")
        exit
            
    # match witness values 
    print("Extracting program from witness...")
    program_nodes = {} # state -> feature
    program_edges = {} # state, partition -> state 
    encoding_file = open(encoding_path,"r")
    for line in encoding_file.readlines():
        if line.startswith("p "):
            break
        if line.startswith("c tau"):
            words = line.split()
            # retrieve var name 
            name = words[1]
            # retrieve var id
            id = words[3]
            if id in var_ids:
                # extract origin state, partition, successor state 
                attributes = name.split('_')
                source  = attributes[1]
                partition = attributes[2]
                index = 2
                while (not partition.isdigit()):
                    index += 1
                    partition = attributes[index]
                index += 1
                dest = attributes[index]
                for att in attributes[index+1:]:
                    dest += "_"+ att
                program_edges[(source,partition)] = dest
                print(f"({source},{partition}) -> {dest}")
        if line.startswith("c lam"):
            words = line.split()
            # retrieve var name 
            name = words[1]
            # retrieve var id
            id = words[3]
            if id in var_ids:
                # extract  state, feature
                attributes = name.split('_')
                node = attributes[1]
                feature = attributes[2]
                for att in attributes[3:]:
                    feature += "_"+att
                program_nodes[node] = feature
                print(f"{node}: {feature}")
    
    witness_file.close()
    return program_edges,program_nodes, count

def synthesize_dot_code(target_path,program_egdes,program_nodes,file_name):
    dot_file_path = target_path+ f"program/program_{file_name}.dot"
    dot_file = open(dot_file_path,"w")

    dot_file.write("digraph {\n")
    for value in program_egdes.values():
        if not value.isdigit():
            dot_file.write(f"node [label={value},style=\"\"] {value}\n")
    
    for node, feature in program_nodes.items():
        dot_file.write(f"node [label={feature},shape=\"diamond\",style=\"\"] {node}\n")

    dot_file.write("\n")

    for (source, partition) , dest in program_egdes.items():
        dot_file.write(f"{source} -> {dest} [label=\"{partition}\"]\n")

    dot_file.write("}")

    dot_file.close()

    png_path = target_path + f"program/program_{file_name}.png"

    os.system(f"dot -Tpng {dot_file_path} -o {png_path}")
    return dot_file_path, png_path

def synthesize_python_code(target_path,program_edges,program_nodes,input_names,file_name):
    python_file_dir_path = target_path+ f"program/"
    program_file_name = f"program_{file_name}.py"
    python_file = open(python_file_dir_path+program_file_name,"w") 

    # target_path_modified = target_path.replace("/",".").rstrip('.')
    python_file.write("import sys\n")
    python_file.write(f"sys.path.insert(0,\"{target_path}\")\n")
    python_file.write(f"import feature_defs\n\n")

    # python_file.write(f"def execute({input_names[0]}")
    # for i in range(1,len(input_names)):
    #     python_file.write(f", {input_names[i]}")
    # python_file.write("):\n")


    python_file.write(f"def execute(inputs):\n")

    # create flowchart edges and nodes maps
    python_file.write(f"\tprogram_nodes ={{}}\n")
    for node,feature in program_nodes.items():
        python_file.write(f"\tprogram_nodes[\"{node}\"]= \"{feature}\"\n")

    python_file.write(f"\n\tprogram_edges ={{}}\n")
    for (source,partition), dest in program_edges.items():
        python_file.write(f"\tprogram_edges[(\"{source}\",{partition})]= \"{dest}\"\n")

    python_file.write("\n\tfeatures = feature_defs.retrieve_feature_defs()\n\n")

    # compute backets for each feature
    input_string = "["
    input_idx = 0
    for name in input_names:
        input_string += f"(\"{name}\",inputs[{input_idx}]),"
        input_idx +=1
    input_string = input_string.rstrip(',')
    input_string += "]"
    
    python_file.write(f"\tvalue_map = {{}} \n")
    for node, feature in program_nodes.items():
        python_file.write(f"\tvalue_map[\"{feature}\"] = features[\"{feature}\"]({input_string})")
        python_file.write("\n")

    python_file.write("\n")
    python_file.write("\tflag = True\n")
    python_file.write("\tcurrent_node = \"0\"\n")
    # python_file.write("\tlabel=\"\"\n")
    python_file.write(f"\twhile flag:\n")
    python_file.write(f"\t\tcurrent_feature = program_nodes[current_node]\n")
    python_file.write(f"\t\tnext_node = program_edges[current_node,value_map[current_feature]]\n")
    python_file.write(f"\t\tif next_node.isdigit():\n")
    python_file.write(f"\t\t\tcurrent_node = next_node\n")
    python_file.write(f"\t\telse:\n")
    python_file.write(f"\t\t\tcurrent_node = next_node\n")
    python_file.write(f"\t\t\tflag = False\n")

    # python_file.write("\tprint(current_node)\n")
    python_file.write(f"\treturn current_node\n\n")

    # python_file.write(f"print(f\"first result {{flowchart(-120.0,32.2,23222.9,1.6)}}\")")


    python_file.close()

    python_file_path = python_file_dir_path+program_file_name

    return python_file_path

def extract_program(encoding_path,witness_path,target_path,input_names,file_name):

    os.system(f"mkdir -p {target_path}program")
    program_edges, program_nodes, count = extract_program_map(encoding_path,witness_path)

    dot_file_path, png_path = synthesize_dot_code(target_path, program_edges, program_nodes,file_name)
    python_file_path = synthesize_python_code(target_path,program_edges,program_nodes,input_names,file_name)

    return python_file_path,dot_file_path,png_path, count

def compute_class_size():
    size = 0

    num_of_features = len(feature_partition)
    max_num_of_partition = 0
    for b in feature_partition.values():
        if b>max_num_of_partition:
            max_num_of_partition=b
    
    num_of_label_combinations = 1
    for b in label_partition.values():
        num_of_label_combinations *=b

    size = num_of_feature_nodes*max_num_of_partition*(num_of_feature_nodes+num_of_label_combinations)*(num_of_features**num_of_feature_nodes)

    # print(f"Class Size:{size}, number of nodes:{num_of_feature_nodes}, feature partitions: {max_num_of_partition}, labels: {num_of_label_combinations}")
    return size

def initialize(benchmark_path):
    
    global num_of_feature_nodes, feature_partition, label_partition, feature_defs, output_path

    print("Extracting synthesis configuration... ") 
    config = configure(benchmark_path) 

    num_of_feature_nodes = config[0]
    feature_partition = config[1]
    label_partition = config[2] 
    feature_defs = config[3]

    output_path = benchmark_path+"mmc_encoding/"


def synthesize(benchmark_path,samples,file_name):
    
    global num_of_feature_nodes, feature_partition, label_partition, feature_defs, output_path

    print(f"Synthesizing program with {num_of_feature_nodes:d} states, features {feature_partition}, and labels {label_partition} using maximum model counting ...")


    # create max#sat encoding
    encoding_file_name = file_name
    encoding_path = encoder.encode(output_path,samples,num_of_feature_nodes,feature_partition,label_partition,feature_defs,encoding_file_name)

    # maximum model counting 
    # print("Maximum model counting...")
    witness_path = output_path + f"{file_name}_witness.txt"
    # os.system(f"python synthesizer/max_sharp_sat/maxcount.py --scalmc synthesizer/max_sharp_sat/scalmc {encoding_path} 1 > {witness_path}")
    run(f"python synthesizer/max_sharp_sat/maxcount.py --scalmc synthesizer/max_sharp_sat/scalmc {encoding_path} 1 > {witness_path}", stdout=PIPE, stderr=STDOUT, universal_newlines=True, shell=True, timeout=300)


    # translate witness to program: extract program from mmc witness
    # extract program signature 
    input_names = []
    for s in samples.keys():
        for i in range(len(s)):
            input_names.append(s[i][0])
        break
    # print(input_names)
    program_path, dot_path, png_path, count  = extract_program(encoding_path,witness_path,benchmark_path,input_names,file_name)
    # os.system(f"open {png_path}")

    # os.system(f"python3 {program_path}")

    return program_path, dot_path, count