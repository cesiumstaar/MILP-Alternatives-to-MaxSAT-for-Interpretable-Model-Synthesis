import math

expl_factor = 1
corr_factor = 1

# Encoding decision diagrams
def phi_E(sat_file,num_of_feature_nodes,feature_partition,label_partition):

    # assign feature nodes to unique features
    def phi_E_feature_det(sat_file,num_of_feature_nodes,feature_partition):
        sat_file.write("phi_E_feature_det :=\n")

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
    def phi_E_transitions(sat_file,num_of_feature_nodes,features_partition,label_partition):
        sat_file.write("phi_E_transitions := \n")

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
    def phi_E_transitions_consistency(sat_file,num_of_feature_nodes,feature_partition,label_partition):
        sat_file.write("phi_E_transitions_consistency :=\n")

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

    phi_E_feature_det(sat_file,num_of_feature_nodes,feature_partition)
    sat_file.write("\n")
    phi_E_transitions(sat_file,num_of_feature_nodes,feature_partition,label_partition)
    sat_file.write("\n")
    phi_E_transitions_consistency(sat_file,num_of_feature_nodes,feature_partition,label_partition)
    sat_file.write("\n")
    sat_file.write("phi_E := phi_E_feature_det & phi_E_transitions & phi_E_transitions_consistency;\n")

# Encoding of relation between samples and decision diagram
def phi_sim(sat_file,num_of_feature_nodes,feature_partition,label_partition,samples,feature_defs):

    # match lables with leaves
    def phi_sim_leaves(sat_file,label_partition,samples,feature_defs):

        # def create_val_string(val):
        #     name = ""
        #     value = val
        #     if val<0:
        #         name += "N"
        #         value = -1*value
        #     else:
        #         name += "P"
            
        #     name += str(int(value*100))
        #     return name

        sat_file.write("phi_sim_leaves :=\n")
        sat_file.write("T")
        count=0
        for s,outputs in samples.items(): 
            for label_name, label_buckets in label_partition.items():
                for bucket in range(label_buckets):
                    sat_file.write(" &\n")
                    sat_file.write(f"(match_{label_name}_{bucket:d}_{count} == ")
                    output_bucket = feature_defs[label_name](outputs)
                    if output_bucket==bucket:
                        sat_file.write("T")
                    else:
                        sat_file.write("F")
                    sat_file.write(")")
                    
                    # sat_file.write("(F")
                    # if output_bucket==bucket:
                    #     sat_file.write(" | ")
                    #     sat_file.write("(T")
                    #     for output_name,output_val in outputs:
                    #         sat_file.write(" & ")
                    #         sat_file.write(f"{output_name}_{create_val_string(output_val)}")
                    #     sat_file.write(")")
                    # sat_file.write(")")
                    # sat_file.write(")")
            count+=1

        sat_file.write(";\n")
    
    
    def phi_sim_transition(sat_file,num_of_feature_nodes,feature_partition,label_partition,samples,feature_defs):
        sat_file.write("phi_sim_transition :=\n")

        sat_file.write("T")
        count=0
        for inputs in samples.keys():
            for node in range(num_of_feature_nodes):
                sat_file.write(" &\n")
                sat_file.write("(")
                sat_file.write(f"match_{node:d}_{count} == ")
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
                            sat_file.write(" | ")
                            sat_file.write("(T")
                            for nodep in range(node+1,num_of_feature_nodes): # for feature nodes
                                sat_file.write(" & ")
                                sat_file.write("(")
                                sat_file.write(f"tau_{node:d}_{feature_bucket:d}_{nodep:d} => match_{nodep:d}_{count}")
                                sat_file.write(")")
                            for label_name, label_buckets in label_partition.items(): # for label nodes
                                for label_bucket in range(label_buckets):
                                    sat_file.write(" & ")
                                    sat_file.write("(")
                                    sat_file.write(f"tau_{node:d}_{feature_bucket:d}_{label_name}_{label_bucket:d} => match_{label_name}_{label_bucket:d}_{count}")
                                    sat_file.write(")")
                                sat_file.write(")")
                                sat_file.write(")")
                                sat_file.write(")")
                sat_file.write(")")
                sat_file.write(")")
            count+=1

        sat_file.write(";\n")

    phi_sim_leaves(sat_file,label_partition,samples,feature_defs)
    sat_file.write("\n")
    phi_sim_transition(sat_file,num_of_feature_nodes,feature_partition,label_partition,samples,feature_defs)
    sat_file.write("\n")
    sat_file.write("phi_sim :=  ")
    sat_file.write("phi_sim_leaves & phi_sim_transition;")
    sat_file.write("\n")

def phi_corr(sat_file,samples):
        sat_file.write("phi_corr := T;")
        # for s in range(len(samples)):
        #     sat_file.write(f" & match_0_{s}")

        #tried this - but gives unsat, I think the corr clauses are being added somewhere else
        # sat_file.write("phi_corr := T")
        # for s in range(len(samples)):
        #     sat_file.write(f" & match_0_{s}")
        # sat_file.write(";\n")


def phi_expl(sat_file,num_of_feature_nodes,feature_partition,label_partition):

    # compute maximum number of buckets
    max_bucket = 0
    for buckets in feature_partition.values():
        if buckets>max_bucket:
            max_bucket = buckets 

    # number of used nodes (reachable nodes)
    sat_file.write("phi_expl := used_node_0")
    for node in range(1,num_of_feature_nodes):
        sat_file.write(f"& ( used_node_{node:d} ==")
        sat_file.write(f"( F")
        for nodep in range(0,node):
            for bucket in range(max_bucket):
                sat_file.write(" | (")
                sat_file.write(f"tau_{nodep:d}_{bucket:d}_{node:d} & used_node_{nodep:d}")
                sat_file.write(")")
        sat_file.write(")")
        sat_file.write(")")

    for node in range(num_of_feature_nodes):
        for feature, bucket in feature_partition.items():
            sat_file.write(" & (")
            sat_file.write(f" lamp_{node:d}_{feature} == (used_node_{node:d} & lam_{node:d}_{feature})")
            sat_file.write(")")
     
    sat_file.write(";\n\n")

def compute_normalizied_weights(feature_weights,num_of_feature_nodes):

    global expl_factor
    
    weights_map = {}
    for node in range(num_of_feature_nodes):
        for feature_name, feature_weight in feature_weights.items():
            weights_map[f"lamp_{node}_{feature_name}"] = feature_weight
    maximum_feature_weight = max(feature_weights.values())
    
    for node in range(num_of_feature_nodes):
        weights_map[f"used_node_{node:d}"] = (maximum_feature_weight+1)     


    weight_sum = sum(weights_map.values())
    for key in weights_map.keys():
        weights_map[key] = int(weights_map[key]/34*100) # TODO implement method to compute max expl: solve maxsat for on phi_expl with its soft clauses 

    return weights_map


def phi_region(sat_file,num_of_feature_nodes,feature_partition,feature_weights,lower_bound,upper_bound, precision):

    weights_map = compute_normalizied_weights(feature_weights,num_of_feature_nodes)

    def enc_in_binary(weight,var_name,precision):
        enc ="T"
        count = 0
        weightp = weight
        while weightp//2 != 0:
            if count>=precision:
                raise Exception(f"Weight not encodable in {precision} bits!")
            
            if weightp%2==1:
                enc = enc + f"& {var_name}_{count}"
            else:
                enc = enc + f"& !{var_name}_{count}"
            count += 1
            weightp = weightp//2

        if weight>0:
            enc = enc + f"& {var_name}_{count}"
            count +=1

        while count<precision:
            enc = enc + f"& !{var_name}_{count}"
            count +=1

        return enc
        
    def phi_exp_weights(sat_file,num_of_feature_nodes,feature_partition,weights_map,precision):
        sat_file.write("phi_exp_weights := T \n")
        for node in range(num_of_feature_nodes):
            sat_file.write("& (")
            sat_file.write(f"!used_node_{node:d} => ")
            sat_file.write("(")
            temp = enc_in_binary(weights_map[f"used_node_{node:d}"],f"b_{node:d}_used",precision)
            sat_file.write(temp)
            sat_file.write(" & ")
            temp = enc_in_binary(0,f"b_{node:d}_lam",precision)
            sat_file.write(temp)
            sat_file.write("))")
            sat_file.write(" & \n")
            sat_file.write("  (")
            sat_file.write(f"used_node_{node:d} =>")
            sat_file.write("( T")
            for bit in range(precision):
                sat_file.write(" & ")
                sat_file.write(f"!b_{node:d}_used_{bit}")
            sat_file.write(")")
            sat_file.write(")")
            for feature, bucket in feature_partition.items():
                sat_file.write("& \n  (")
                sat_file.write(f"lamp_{node:d}_{feature} => ")
                sat_file.write("(")
                temp = enc_in_binary(weights_map[f"lamp_{node:d}_{feature}"], f"b_{node:d}_lam", precision)
                sat_file.write(temp)
                sat_file.write(")")
                sat_file.write(")")
            sat_file.write("\n")
        sat_file.write(";\n\n")

    def add(goal_var_name, left_var_name, right_var_name, precision):
        enc = f"!c_{goal_var_name}_0  "
        for bit in range(precision):
            enc += f"& ({goal_var_name}_{bit} == ((!c_{goal_var_name}_{bit} & ({left_var_name}_{bit}^{right_var_name}_{bit})) | (c_{goal_var_name}_{bit} & ({left_var_name}_{bit} == {right_var_name}_{bit}))))"
        for bit in range(1,precision):
            enc += f"& (c_{goal_var_name}_{bit} == ((!c_{goal_var_name}_{bit-1} & {left_var_name}_{bit-1} & {right_var_name}_{bit-1}) | (c_{goal_var_name}_{bit-1} &({left_var_name}_{bit-1} | {right_var_name}_{bit-1}))))"
        return enc

    def phi_adder(sat_file,num_of_feature_nodes,precision):
        sat_file.write("phi_adder := T \n")
        for bit in range(precision):
            sat_file.write(f"& !a_0_used_{bit} & !a_0_lam_{bit}")
        for node in range(num_of_feature_nodes):
            sat_file.write(" & (")
            temp = add(f"a_{node+1:d}_used", f"a_{node:d}_used", f"b_{node:d}_used", precision)
            sat_file.write(temp)
            sat_file.write(") \n")
        for node in range(num_of_feature_nodes):
            sat_file.write(" & (")
            temp = add(f"a_{node+1:d}_lam", f"a_{node:d}_lam", f"b_{node:d}_lam", precision)
            sat_file.write(temp)
            sat_file.write(")\n")
        sat_file.write(" & (")
        temp = add(f"a_fin", f"a_{num_of_feature_nodes:d}_used", f"a_{num_of_feature_nodes:d}_lam", precision)
        sat_file.write(temp)
        sat_file.write(")\n")
        sat_file.write(";\n\n")

    def phi_threshold(sat_file,lower_bound,upper_bound,precision, lower, upper):
        sat_file.write(f"phi_threshold := T ")
        
        if upper:
            sat_file.write("&\n")
            sat_file.write(f"smaller_{precision-1}")
            for bit in range(1,precision):
                sat_file.write(f"& (smaller_{bit} == ((!a_fin_{bit} & upper_{bit})| ( (a_fin_{bit} == upper_{bit}) & smaller_{bit-1})))")
            sat_file.write(f"& (smaller_0 == (!a_fin_0 & upper_0))\n")
       
        if lower:
            sat_file.write("&\n")
            sat_file.write(f"larger_{precision-1}")
            for bit in range(1,precision):
                sat_file.write(f"& (larger_{bit} == ((a_fin_{bit} & !lower_{bit})| ((a_fin_{bit} == lower_{bit}) & larger_{bit-1})))")
            sat_file.write(f"& (larger_0 == (a_fin_0 & !lower_0))\n")

        if upper:
            sat_file.write("& \n")
            temp = enc_in_binary(upper_bound, "upper", precision)
            sat_file.write(temp)
        if lower:
            sat_file.write("& \n")
            temp = enc_in_binary(lower_bound, "lower", precision)
            sat_file.write(temp)
        sat_file.write(";\n\n")

    if lower_bound>0 or upper_bound<100:
        # Encode weights of syntactic structures
        phi_exp_weights(sat_file,num_of_feature_nodes,feature_partition,weights_map,precision)
        # Encode adder for summing up weights
        phi_adder(sat_file,num_of_feature_nodes,precision)
        phi_threshold(sat_file,lower_bound,upper_bound,precision, lower_bound>0, upper_bound<100)
        sat_file.write("phi_region := phi_exp_weights & phi_adder & phi_threshold;\n\n ")
    else:
        sat_file.write("phi_region := T;\n\n")

def extract_threshold_vars(dimacs):
    print("|---Extracting threshold variables...") 

    threshold_vars = {}
    text_file = open(dimacs)

    for line in text_file.readlines():
        if not line.startswith("c"):
            break

        current_num = ""
        current_var = ""

        if ("upper_" in line or "lower_" in line):
            words = line.split()
            current_var = words[1]
            current_num = words[3]

            threshold_vars[current_var]= current_num

    return threshold_vars

def extract_soft_variables(dimacs,feature_weights,num_of_feature_nodes):
    print("|---Extracting soft variables...")  
    soft_vars = []
    expl_soft_vars = []
    corr_soft_vars = []
    corr_vars = []
    text_file = open(dimacs)
    

    weights_map = compute_normalizied_weights(feature_weights,num_of_feature_nodes)

    for line in text_file.readlines():
        if not line.startswith("c"):
            break
        
        current_num = ""
        current_var = ""
        if ("match_0" in line or " lamp_" in line or "used_node_" in line):
            words = line.split()
            current_var = words[1]
            current_num = words[3]

            if current_var in weights_map.keys():
                weight = int(weights_map[current_var]/corr_factor)
                if "used_node_" in current_var:
                    if((f"-{current_num}",weight) not in soft_vars):
                        soft_vars.append((f"-{current_num}",weight))
                        expl_soft_vars.append((f"-{current_num}",weight))
                if "lamp_" in current_var:
                    if((current_num,weight) not in soft_vars):
                        soft_vars.append((current_num,weight))
                        expl_soft_vars.append((current_num,weight))             
            else:
                weight = 1
                if((current_num,weight) not in soft_vars):
                        soft_vars.append((current_num,weight))
                        corr_soft_vars.append((current_num,weight))
                corr_vars.append((current_num,weight))


            # reached_num = False
            # finished_var = False
            # for letter in line:
            #     if letter=='\n':
            #         break
            #     if reached_num:
            #         if letter.isdigit():
            #             current_num += letter
            #     if letter == '>':
            #         reached_num = True
        
    text_file.close()
    return soft_vars, corr_soft_vars, expl_soft_vars, corr_vars
# def phi_soft(sat_file, samples, num_of_feature_nodes,feature_partition):
#     sat_file.write("phi_soft := ")
#     phi_corr(sat_file,samples)
#     for node in range(num_of_feature_nodes):
#         sat_file.write(f"& !used_node_{node:d}")
#     for node in range(num_of_feature_nodes):
#         for feature, buckets in feature_partition.items():
#             sat_file.write(f" & lam_{node:d}_{feature}") 
#     sat_file.write(";\n\n")