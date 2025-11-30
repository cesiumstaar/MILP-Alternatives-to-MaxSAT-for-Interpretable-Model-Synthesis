import os



def dimacs_to_wcnf(wcnf_file, dimacs_path, soft_vars):
    print("|---Writing WCNF File...")  
    text_file = open(dimacs_path)

    # print(pi_vars)
    hard_limit = 1
    for num, weight in soft_vars:
        hard_limit += int(weight)

    for line in text_file.readlines():
        if  line.startswith("c"):
            wcnf_file.write(line)
            continue
           
        if line.startswith('p cnf'):
            #print (line)
            var = line.split(' ')[2]
            numClauses=int(line.split(' ')[3])
            # print (var, numClauses)
            wcnf_file.write("p wcnf ")
            wcnf_file.write(var)
            wcnf_file.write(" ")
            wcnf_file.write(str(numClauses+ len(soft_vars))) 
            wcnf_file.write(f" {hard_limit}  \n")  #limit for hard clauses
            # add soft clauses
            for var,weight in soft_vars:
                wcnf_file.write(f"{weight} {var} {0}\n")
        else: 
            # words = line.split()
            # first_var = words[0]
            # second_var = words[1]
            # if ((first_var in soft_vars) and (int(second_var) == 0)):
            #     wcnf_file.write(f"1 {first_var} {second_var}\n") 
            # else:
            wcnf_file.write(f"{hard_limit} {line}")

    text_file.close()    
    return hard_limit





def encode(encoder, output_dir_path,samples,num_of_feature_nodes,feature_partition,feature_weights,label_partition,feature_defs,lower_bound, upper_bound, precision, weights_map, file_name):

    # create max#sat file
    os.system(f"mkdir -p {output_dir_path}")
    sat_file_path = output_dir_path+file_name
    sat_file = open(sat_file_path,"w")
    sat_file.write("BC1.1\n")

    # phi_E
    encoder.phi_E(sat_file,num_of_feature_nodes,feature_partition,label_partition)
    sat_file.write("\n")

    # phi_sim 
    encoder.phi_sim(sat_file,num_of_feature_nodes,feature_partition,label_partition,samples,feature_defs)
    sat_file.write("\n")

    # phi_corr
    encoder.phi_corr(sat_file,samples)
    sat_file.write("\n") 

    # phi_expl
    encoder.phi_expl(sat_file,num_of_feature_nodes,feature_partition,label_partition)
    sat_file.write("\n")   

    # phi_region
    encoder.phi_region(sat_file,num_of_feature_nodes,feature_partition,feature_weights,lower_bound,upper_bound,precision)


    sat_file.write("FORMULA :=  phi_E & phi_sim & phi_corr & phi_expl & phi_region;\n")
    

    sat_file.write("ASSIGN FORMULA;")
    sat_file.close()


    # create dimacs file
    dimacs_path = output_dir_path+"encoding_temp.dimacs"

    print("Translating to dimacs...")
    os.system(f"./synthesizer/max_sat/bc2cnf -all  {sat_file_path} {dimacs_path}")


    print("Translating to wcnfdimacs...")
    wcnf_path = output_dir_path+file_name+".wcnf"
    soft_vars, corr_soft_vars, expl_soft_vars, corr_vars = encoder.extract_soft_variables(dimacs_path,feature_weights,num_of_feature_nodes)#TODO extract should only take dimacs path. Every other info should be in encoder
    threshold_vars = encoder.extract_threshold_vars(dimacs_path)
    wcnf_file = open(wcnf_path,"w")
    hard_weight = dimacs_to_wcnf(wcnf_file, dimacs_path, soft_vars)
    wcnf_file.close()

    return wcnf_path, corr_soft_vars, expl_soft_vars, corr_vars, threshold_vars, hard_weight

