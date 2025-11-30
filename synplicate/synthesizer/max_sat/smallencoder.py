import os
#from pysat.examples.fm import FM

## Code same as that written by Hazem. Made minor modifications in the phi_sim functions.

def create_val_string(val):
    name = ""
    value = val
    if val<0:
        name += "N"
        value = -1*value
    else:
        name += "P"
    
    name += str(int(value*100))
    return name

def phi_Gamma_val(sat_file,samples):
    val_var_map = {} # maps input number and sample number to variable 

    #sat_file.write("phi_Gamma_val := \n")
    #sat_file.write("phi_Gamma :=  \n")
    sat_file.write("phi_Gamma := T; \n")
    #sat_file.write("F")
    for inputs, labels in samples.items():
      #  sat_file.write("|\n")
      #  sat_file.write("( T")
        for inp in inputs:
            if inp[0] not in val_var_map.keys():
                val_var_map[inp[0]] = []
            temp_string = f"{inp[0]}_{create_val_string(inp[1])}"
      #      sat_file.write(f" & {temp_string}")
            if temp_string not in val_var_map[inp[0]]:
                val_var_map[inp[0]].append(temp_string)
        for label in labels:
            if label[0] not in val_var_map.keys():
                val_var_map[label[0]] = []
            temp_string = f"{label[0]}_{create_val_string(label[1])}"
        #   sat_file.write(f" & {temp_string}")
            if temp_string not in val_var_map[label[0]]:
                val_var_map[label[0]].append(temp_string)
    #    sat_file.write(")")

    #sat_file.write(";\n")
    return val_var_map

#def phi_Gamma_det(sat_file,samples,val_var_map):
#    sat_file.write("phi_Gamma_det := \n")

#    sat_file.write("T")
#    for name, values in val_var_map.items():
#        sat_file.write(" &\n")
#        sat_file.write(f"(T")
#        for i in range(len(values)):
#            sat_file.write(" & (")
#            sat_file.write(f"{values[i]} => ")
#            sat_file.write("(T")
#            for j in range(i+1,len(values)):
#                sat_file.write(" & ")
#                sat_file.write(f"!{values[j]}")
#            sat_file.write(")")
#            sat_file.write(")")
#        sat_file.write(")")

#    sat_file.write(";\n")

def phi_Gamma(sat_file,samples):
    val_var_map ={} # maps input number and sample number to variable 
    
    val_var_map = phi_Gamma_val(sat_file,samples) 
    sat_file.write("\n")
    #    phi_Gamma_det(sat_file,samples,val_var_map)
    sat_file.write("\n")
    #      sat_file.write("phi_Gamma :=  phi_Gamma_val & phi_Gamma_det;\n")
   
# assign feature nodes to unique features
def phi_T_feature_det(sat_file,num_of_feature_nodes,feature_partition):
    sat_file.write("phi_T_feature_det :=\n")

    sat_file.write("T")
    for node in range(num_of_feature_nodes):
        sat_file.write(" &\n")
        sat_file.write("(F")
        for feature in feature_partition.keys():
            sat_file.write(" | ")
            sat_file.write("(")
            sat_file.write(f"lam_{node:d}_{feature}")
            for featurep in feature_partition.keys():
                if featurep!=feature:
                    sat_file.write(" & ")
                    sat_file.write(f"!lam_{node:d}_{featurep}")
            sat_file.write(")")
        sat_file.write(")")
    sat_file.write(";\n")

# transition relation of template
def phi_T_transitions(sat_file,num_of_feature_nodes,features_partition,label_partition):
    sat_file.write("phi_T_transitions := \n")

    sat_file.write("T")
    for node in range(num_of_feature_nodes):
        for feature, feature_buckets in features_partition.items():
            if feature_buckets > 0:
                sat_file.write(" & \n")
                sat_file.write("(")
                sat_file.write(f"lam_{node:d}_{feature} =>\n")
                sat_file.write("          (T")
                for feature_bucket in range(feature_buckets):
                    sat_file.write(" &\n")
                    sat_file.write("          (F")
                    for nodep in range(node+1,num_of_feature_nodes):
                        sat_file.write(" | ")
                        sat_file.write("(")
                        sat_file.write(f"tau_{node:d}_{feature_bucket:d}_{nodep} &")
                        sat_file.write("(T")
                        for nodepp in range(node+1,num_of_feature_nodes):
                            if nodepp != nodep:
                                sat_file.write(" & ")
                                sat_file.write(f"!tau_{node:d}_{feature_bucket:d}_{nodepp}")
                        for label, label_buckets in label_partition.items():
                            for label_bucket in range(label_buckets):
                                sat_file.write(" & ")
                                sat_file.write(f"!tau_{node:d}_{feature_bucket:d}_{label}_{label_bucket}")
                        sat_file.write(")")
                        sat_file.write(")")
                    sat_file.write(" | (T")
                    for label, label_buckets in label_partition.items():
                        sat_file.write(" & ")
                        sat_file.write("(F")
                        for label_bucket in range(label_buckets):
                            sat_file.write(" | ")
                            sat_file.write("(")
                            sat_file.write(f"tau_{node:d}_{feature_bucket:d}_{label}_{label_bucket:d} & ")
                            sat_file.write("(T")
                            for label_bucketp in range(label_buckets):
                                if label_bucketp != label_bucket:
                                    sat_file.write(" & ")
                                    sat_file.write(f"!tau_{node:d}_{feature_bucket:d}_{label}_{label_bucketp}")
                            for nodep in range(node+1,num_of_feature_nodes):
                                sat_file.write(" & ")
                                sat_file.write(f"!tau_{node:d}_{feature_bucket:d}_{nodep}")
                            sat_file.write(")")
                            sat_file.write(")")

                        sat_file.write(")")  
            
                    sat_file.write(")")
                    sat_file.write(")")
                sat_file.write(")")
                sat_file.write(")")
    sat_file.write(";\n")

# only buckets of feature in node are allowed
def phi_T_transitions_consistency(sat_file,num_of_feature_nodes,feature_partition,label_partition):
    sat_file.write("phi_T_transitions_consistency :=\n")

    # compute maximum number of buckets
    max_bucket = 0
    for buckets in feature_partition.values():
        if buckets>max_bucket:
            max_bucket = buckets 

    sat_file.write("T")
    for node in range(num_of_feature_nodes):
        for feature, feature_buckets in feature_partition.items():
            sat_file.write(" &\n")
            sat_file.write("(")
            sat_file.write(f"lam_{node:d}_{feature} => ")
            sat_file.write("(T")
            for feature_bucket in range(feature_buckets,max_bucket):
                for nodep in range(node+1,num_of_feature_nodes):
                    sat_file.write(" & ")
                    sat_file.write(f"!tau_{node}_{feature_bucket:d}_{nodep:d}")
                for label,label_buckets in label_partition.items():
                    for label_bucket in range(label_buckets):
                        sat_file.write(" & ")
                        sat_file.write(f"!tau_{node}_{feature_bucket:d}_{label}_{label_bucket:d}")
            sat_file.write(")")
            sat_file.write(")")
    sat_file.write(";\n")

def phi_T(sat_file,num_of_feature_nodes,feature_partition,label_partition):

    phi_T_feature_det(sat_file,num_of_feature_nodes,feature_partition)
    sat_file.write("\n")
    phi_T_transitions(sat_file,num_of_feature_nodes,feature_partition,label_partition)
    sat_file.write("\n")
    phi_T_transitions_consistency(sat_file,num_of_feature_nodes,feature_partition,label_partition)
    sat_file.write("\n")
    sat_file.write("phi_T := phi_T_feature_det & phi_T_transitions & phi_T_transitions_consistency;\n")

def phi_sim_leaves(sat_file,label_partition,samples,feature_defs):
    sat_file.write("phi_sim_leaves :=\n")

    sat_file.write("T")
    count=0
    for s,outputs in samples.items(): 
        for label_name, label_buckets in label_partition.items():
            for bucket in range(label_buckets):
                sat_file.write(" &\n")
                sat_file.write(f"(alp_{label_name}_{bucket:d}_{count} == ")
                sat_file.write("(F")
            #    for outputs in o:
                output_bucket = feature_defs[label_name](outputs)
                if output_bucket==bucket:
                    sat_file.write(" | ")
                    sat_file.write("(T")
                    for output_name,output_val in outputs:
                        sat_file.write(" & ")
                        sat_file.write(f"{output_name}_{create_val_string(output_val)}")
                    sat_file.write(")")
                sat_file.write(")")
                sat_file.write(")")
        count+=1;

    sat_file.write(";\n")

def phi_sim_transition(sat_file,num_of_feature_nodes,feature_partition,label_partition,samples,feature_defs):
    sat_file.write("phi_sim_transition :=\n")

    sat_file.write("T")
    count=0
    for inputs in samples.keys():
        for node in range(num_of_feature_nodes):
            sat_file.write(" &\n")
            sat_file.write("(")
            sat_file.write(f"pi_{node:d}_{count} == ")
            sat_file.write("(F")
            for feature, feature_buckets in feature_partition.items():
                for feature_bucket in range(feature_buckets):
                    input_bucket = feature_defs[feature](inputs)
                    #print ({input_bucket}, {feature_bucket}, {features}, {inputs})
                    if input_bucket==feature_bucket:
                        sat_file.write(" | \n")
                        sat_file.write("         (")
                        sat_file.write(f"lam_{node:d}_{feature}") # feature in node 
                        sat_file.write(" & (F") # compare to chosen sample
               #     for inputs in s.keys():
                        sat_file.write(" | ")
                        sat_file.write("(T")
                        #for input_name, input_val in inputs:
                        #    sat_file.write(" & ")
                        #    sat_file.write(f"{input_name}_{create_val_string(input_val)}")
                    #   #     sat_file.write(")")
                        #sat_file.write(")")
                        #sat_file.write(" & (T") # transitions with feature_bucket
                        for nodep in range(node+1,num_of_feature_nodes): # for feature nodes
                            sat_file.write(" & ")
                            sat_file.write("(")
                            sat_file.write(f"tau_{node:d}_{feature_bucket:d}_{nodep:d} => pi_{nodep:d}_{count}")
                            sat_file.write(")")
                        for label_name, label_buckets in label_partition.items(): # for label nodes
                            for label_bucket in range(label_buckets):
                                sat_file.write(" & ")
                                sat_file.write("(")
                                sat_file.write(f"tau_{node:d}_{feature_bucket:d}_{label_name}_{label_bucket:d} => alp_{label_name}_{label_bucket:d}_{count}")
                                sat_file.write(")")
                            sat_file.write(")")
                            sat_file.write(")")
                            sat_file.write(")")
            sat_file.write(")")
            sat_file.write(")")
        count+=1

    sat_file.write(";\n")

# path encoding
def phi_sim(sat_file,num_of_feature_nodes,feature_partition,label_partition,samples,feature_defs):

    phi_sim_leaves(sat_file,label_partition,samples,feature_defs)
    sat_file.write("\n")
    phi_sim_transition(sat_file,num_of_feature_nodes,feature_partition,label_partition,samples,feature_defs)
    sat_file.write("\n")
    sat_file.write("phi_sim := ");
    for s in range(len(samples)):
        sat_file.write(f"pi_0_{s} & ");
    sat_file.write( " phi_sim_leaves & phi_sim_transition;\n")

def write_all_clauses(dimacs_file, dimacs_tempfile_path, pi_vars):
    print("|---Writing WCNF File...")  
    text_file = open(dimacs_tempfile_path)

    # print(pi_vars)
    hard_limit = len(pi_vars) + 1
    for line in text_file.readlines():
        if  line.startswith("c"):
            dimacs_file.write(line)
            continue
           
        if line.startswith('p cnf'):
            #print (line)
            var = line.split(' ')[2]
            numClauses=int(line.split(' ')[3])
            # print (var, numClauses)
            dimacs_file.write("p wcnf ")
            dimacs_file.write(var)
            dimacs_file.write(" ")
            dimacs_file.write(str(numClauses + len(pi_vars)))
            dimacs_file.write(f" {hard_limit}  \n");   #limit for hard clauses
        else: 
            words = line.split()
            first_var = words[0]
            second_var = words[1]
            if ((first_var in pi_vars) and (int(second_var) == 0)):
                dimacs_file.write(f"1 {first_var} {second_var}\n") 
            else:
                dimacs_file.write(f"{hard_limit} {line}")
    #for s in pi_vars:
        #dimacs_file.write(f"1 {s} 0\n")
    text_file.close()    
    
def extract_pi_max_variables(dimacs):
    print("|---Extracting alpha (pi)  variables...")  
    pi_vars = []
    #tau_vars = []
    #lam_vars = []
    text_file = open(dimacs)

    for line in text_file.readlines():
        if not line.startswith("c"):
            break
        
        current_num = ""
       
        if ("pi_0" in line ): # or "tau" in line or "lam" in line ):
            flag = False
            
            for letter in line:
                if letter=='\n':
                    break
                if flag:
                    if letter.isdigit():
                        current_num += letter
                if letter == '>':
                    flag = True
        
        if current_num!="":
        #   if ("pi" in line):
            pi_vars.append(current_num)
        #    if("tau" in line):
            #    tau_vars.append(current_num)
        #    if ("lam" in line):
            #    lam_vars.append(current_num)

    # print(f"   {len(pi_vars)} pi variables")
    # print(f"   {len(tau_vars)} tau variables")
    # print(f"   {len(lam_vars)} lam variables")
    text_file.close()    
    #return pi_vars, tau_vars, lam_vars
    return pi_vars #, tau_vars, lam_vars

def extract_ind_variables(dimacs,name_list): 
    print("|---Extracting counting variables...")   
    vars = []
    text_file = open(dimacs)

    for line in text_file.readlines():
        # print(line)
        if not line.startswith("c"):
            break
        
        current_num = ""
        flag = False
        for name in name_list:
            if (f"c {name}") in line:
                # print(name)
                flag = True
       
        if flag:
            flag2= False
            for letter in line:
                if letter=='\n':
                    break
                if flag2:
                    if letter.isdigit():
                        current_num += letter
                if letter == '>':
                    flag2 = True
        
        if current_num!="":
            vars.append(current_num)
    # print(f"   {len(vars)} ind variables")
    text_file.close()
    return vars



def encode(output_path,samples,num_of_feature_nodes,feature_partition,label_partition,feature_defs,file_name):

    # create max#sat file
    os.system(f"mkdir -p {output_path}")
    sat_file_path = output_path+file_name
    sat_file = open(sat_file_path,"w")
    sat_file.write("BC1.1\n")

    # phi_E
    phi_T(sat_file,num_of_feature_nodes,feature_partition,label_partition)
    sat_file.write("\n")
    
    # phi_sim 
    # compute mapping between (feature,bucket) -> sample
    phi_sim(sat_file,num_of_feature_nodes,feature_partition,label_partition,samples,feature_defs)
    sat_file.write("\n")

    sat_file.write("FORMULA := phi_Gamma & phi_T & phi_sim;\n")
    sat_file.write("ASSIGN FORMULA;")
    sat_file.close()

    # create dimac file with mmc parameters 
    dimacs_tempfile_path = output_path+"encoding_temp.dimacs"

    print("Translating to dimacs...")
    os.system(f"./synthesizer/max_sat/bc2cnf -all {sat_file_path} {dimacs_tempfile_path}")
    

    print("Translating to maxdimacs...")
    dimacs_file_path = output_path+file_name+".wcnf"
    pi_vars = extract_pi_max_variables(dimacs_tempfile_path)

    #dimacs_file.write("c max")
    #for s in maximization_vars:
    #    dimacs_file.write(f" {s} ")
    #dimacs_file.write("0\n")    

    #dimacs_file.write("c ind")

    dimacs_file = open(dimacs_file_path,"w")
    write_all_clauses(dimacs_file, dimacs_tempfile_path, pi_vars)


#    dimacs_temp_file = open(dimacs_tempfile_path,"r")
#    temp_string = dimacs_temp_file.read()
#    dimacs_file.write(temp_string)
#    dimacs_temp_file.close()
    dimacs_file.close()

    #os.system(f"rm {output_path}encoding_temp.dimacs")

    return dimacs_file_path,pi_vars
