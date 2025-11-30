import os
import importlib
import sys
sys.path.append('../../')
sys.path.append('../')
from subprocess import STDOUT, PIPE, run

#import procedures
from methods.erm import erm_procedure as erm
from methods.cegqs import cegqs_procedure as cegqs

# import synthesizers
from synthesizer.max_sharp_sat import mmc_synthesizer
from synthesizer.max_sat import ms_synthesizer
from synthesizer.omt import omt_synthesizer
from synthesizer.decision_trees import cart_tree_synthesizer

# from evaluator.recall_evaluator import evaluator
from evaluator.cochran_evaluator import evaluator 

import timeit





mmc_flag = False

def experiment(dd_size, delta, epsilon, dir_name,depth, num_iter,num_eva):

    # synthesis_benchmark_path = f"experiments/ICML/loan_acquisition/"
    synthesis_benchmark_path = f"experiments/ICML/AutoTaxi/"
    # synthesis_benchmark_path = f"experiments/ICML/theorem_prover/"
    # synthesis_benchmark_path = f"experiments/ICML/california_census/"


    os.system(f"rm -r {synthesis_benchmark_path}program")
    os.system(f"rm -r {synthesis_benchmark_path}samples")
    test = synthesis_benchmark_path.replace("/",".").rstrip('.')

    ################################################################################################################
    #                                    GLOBAL PERFORMANCE-PARAMETER VARIABLES                                    #
    ################################################################################################################
    
    #TIME
    erm_synthesis_time = 0
    erm_mmc_synthesis_time = 0
    
    cart_depth_synthesis_time = 0
    cart_unbounded_synthesis_time = 0
    
    inferdt_synthesis_time = 0
    inferdt_depth_synthesis_time = 0
    binoct_synthesis_time = 0
   
    cegqs_synthesis_time = 0
    cegqs_mmc_synthesis_time = 0
    

    ################################################################################################################
    #                         SYNTHESIZE USING SYNPLICATE-ERM, CART, INFERDT, AND BINOCT                           #
    ################################################################################################################

    # SYNTHESIZE USING SYNPLICTAE AND USE PRODUCED SAMPLES FOR INFERDT AND BINOCT
    print("Executing Synplicate's erm procedure ...")
    # USING MS
    start = timeit.default_timer()
    erm_samples, erm_program_path = erm.execute(synthesis_benchmark_path,ms_synthesizer,delta,epsilon,"ms")
    stop = timeit.default_timer()
    erm_synthesis_time = stop-start

    samples_file = f"{synthesis_benchmark_path}samples/erm_syn_samples.csv"

    # USING MMC
    if mmc_flag:
        start = timeit.default_timer()
        erm_mmc_samples, erm_mmc_program_path = erm.execute(synthesis_benchmark_path,mmc_synthesizer,delta,epsilon,"mmc")
        stop = timeit.default_timer()
        erm_mmc_synthesis_time = stop-start

    
    
    # USING INFERDT
    ## AUX
    def parse_inferdt(tree_text,out_file_path):
        program = open(out_file_path,"w")
        tokens  = (tree_text.stdout).split("\n")

        indent = "   "
        program.write("def execute(in_map):\n")
        depth = 0
        nodes = 0
        leaves = 0
        for token in tokens:
            if "Depth:" in token:
                temp = token.split(":")
                depth = int(temp[1])
            if "Number of nodes:" in token:
                temp = token.split(":")
                nodes = int(temp[1])
            if ">" in token:
                tabs = token.count("|")+1
                temp = token.replace("|","").strip().split(">")
                if not "!" in token:
                    if ":" in temp[1]:
                        tempp = temp[1].split(":")
                        program.write(f"{indent*tabs}if in_map['{temp[0]}']>{tempp[0]}:\n{indent*(tabs+1)}return {tempp[1]}\n")
                        leaves +=1
                    else:    
                        program.write(f"{indent*tabs}if in_map['{temp[0]}']>{temp[1]}:\n")
                else:
                    temp = token.replace("!","").replace("|","").strip().split(">")
                    if ":" in temp[1]:
                        tempp = temp[1].split(":")
                        program.write(f"{indent*tabs}else:\n{indent*(tabs+1)}return {tempp[1]}\n")
                        leaves += 1
                    else:    
                        program.write(f"{indent*tabs}else:\n")

    
        program.close()
        return depth, nodes, leaves, out_file_path

    os.system(f"mkdir {synthesis_benchmark_path}inferdt")
    ## OPTIMAL DEPTH AND SIZE
    print("Executing InferDT depth + size ...")
    timeout_flag = False
    start = timeit.default_timer()
    try:
        inferdt_depth_size_tree = run(f"./synthesizer/InferDT/InferDT {samples_file} -v -x infer", stdout=PIPE, stderr=STDOUT, universal_newlines=True, shell=True, timeout=300)
        inferdt_depth_size_depth, inferdt_depth_size_nodes, inferdt_depth_size_leaves, inferdt_depth_size_path = parse_inferdt(inferdt_depth_size_tree,f"{synthesis_benchmark_path}inferdt/dt_d_s.py")
        print(inferdt_depth_size_depth,inferdt_depth_size_nodes, inferdt_depth_size_leaves)
    except:
        print("InferDT timeout!")
        timeout_flag = True
        inferdt_depth_size_depth = "N/A"
        inferdt_depth_size_nodes = "N/A"
        inferdt_depth_size_leaves = "N/A"
    stop = timeit.default_timer()
    if timeout_flag:
        inferdt_synthesis_time = "TO"
    else:
        inferdt_synthesis_time = stop-start
    

    ## OPTIMAL DEPTH
    print("Executing InferDT depth ...")
    timeout_flag = False
    start = timeit.default_timer()
    try:
        inferdt_depth_tree = run(f"./synthesizer/InferDT/InferDT {samples_file} -d -v -x infer", stdout=PIPE, stderr=STDOUT, universal_newlines=True, shell=True, timeout=300)
        inferdt_depth_depth, inferdt_depth_nodes, inferdt_depth_leaves, inferdt_depth_path = parse_inferdt(inferdt_depth_tree,f"{synthesis_benchmark_path}inferdt/dt_d.py")
        print(inferdt_depth_depth,inferdt_depth_nodes,inferdt_depth_leaves)
    except:
        print("InferDT Timeout")
        timeout_flag = True
        inferdt_depth_depth = "N/A"
        inferdt_depth_nodes = "N/A"
        inferdt_depth_leaves = "N/A"
    stop = timeit.default_timer()
    if timeout_flag:
        inferdt_depth_synthesis_time = "TO"
    else:
        inferdt_depth_synthesis_time = stop-start
    


    # USING BINOCT
    # AUX
    def parse_binoct(tree_text,out_file_path):
        program = open(out_file_path,"w")
        tokens  = (tree_text.stdout).split("\n")

        
        program.write("def execute(in_map):\n")
        nodes = 0
        leaves = 0
        for token in tokens:
            if "if" in token:
                temp = token.replace("{",":").split(" <= ")
                tempp = temp[0].split("( ")
                program.write("\n   ")
                program.write(f"{tempp[0]} ( in_map['{tempp[1]}'] <= {temp[1]}")
                nodes += 1
            if "examples" in token:
                temp = token.split("return")
                tempp = temp[1].split()
                program.write("\n   ")
                program.write(f"{temp[0]} return {tempp[0]}")
                nodes += 1
                leaves += 1
            if "else" in token:
                temp = token.replace("{",":")
                program.write("\n   ")
                program.write(temp)
            

        program.close()
        return nodes, leaves, out_file_path
    binoct_depth = 10
    if inferdt_depth_depth != "N/A":
        binoct_depth = inferdt_depth_depth
    os.system(f"mkdir {synthesis_benchmark_path}binoct")
    # USES DEPTH FORM INFERDT DEPTH
    print("Executing BinOct with maxdepth from InferDT ...")
    timeout_flag = False
    start = timeit.default_timer()
    try:
        binoct_tree = run(f"python3 ./synthesizer/binoct/run_exp.py {samples_file} {binoct_depth}",stdout=PIPE, stderr=STDOUT, universal_newlines=True, shell=True, timeout=300)
        binoct_nodes, binoct_leaves, binoct_path = parse_binoct(binoct_tree,f"{synthesis_benchmark_path}binoct/dt.py")
        print(binoct_tree.stdout)
        print("Number of nodes and leaves for binoct tree: ",binoct_nodes, binoct_leaves)
    except:
        print("BinOct Timeout")
        timeout_flag = True
        binoct_path = "N/A"
        binoct_leaves = "N/A"
        binoct_nodes = "N/A"
        binoct_depth = "N/A"
    stop = timeit.default_timer()
    if timeout_flag:
        binoct_synthesis_time = "TO"
    else:
        binoct_synthesis_time = stop-start
    

    
    # USING SKLEARN CART
    print("Executing CART unbounded depth ...")
    ## UNBOUNDED DEPTH 
    start = timeit.default_timer()
    cart_unbounded_tree = cart_tree_synthesizer.synthesize(samples_file,["0","1","2","3"], synthesis_benchmark_path,-1,"unbounded")
    cart_unbounded_depth = cart_unbounded_tree.get_depth()
    cart_unbounded_nodes = cart_unbounded_tree.get_n_leaves()
    stop = timeit.default_timer()
    cart_unbounded_synthesis_time = stop-start

    ## BOUNDED DEPTH 
    cart_depth = 10
    if binoct_depth!= "N/A":
        cart_depth = binoct_depth
    print("Executing CART bounded depth ...")
    start = timeit.default_timer()
    cart_depth_tree = cart_tree_synthesizer.synthesize(samples_file,["0","1","2","3"], synthesis_benchmark_path,cart_depth,"depth")
    cart_bounded_depth = cart_depth_tree.get_depth()
    cart_bounded_nodes = cart_depth_tree.get_n_leaves()
    stop = timeit.default_timer()
    cart_depth_synthesis_time = stop-start
    ################################################################################################################
    #                                       SYNTHESIZE USING SYNPLICATE-CEGS                                       #
    ################################################################################################################
    
    # USING MMC
    if mmc_flag:
        start = timeit.default_timer()
        cegqs_mmc_program_path, cegqs_best_mmc  = cegqs.execute(synthesis_benchmark_path,mmc_synthesizer, delta, epsilon, 10, num_iter, 0.01, 15, "ttf", 15,"mmc")
        stop = timeit.default_timer()
        cegqs_mmc_synthesis_time = stop-start
    else:
        cegqs_best_mmc = "N/A"

    # USING MS
    start = timeit.default_timer()
    cegqs_program_path, cegqs_best_ms  = cegqs.execute(synthesis_benchmark_path,ms_synthesizer, delta, epsilon, 10, num_iter, 0.01, 15, "ttf", 15,"ms")
    stop = timeit.default_timer()
    cegqs_synthesis_time = stop-start


    ################################################################################################################

    ################################################################################################################
    #                                                   EVALUATION                                                 #
    ################################################################################################################

    # GLOBAL VARIBALES FOR AVERAGE STATISTICS
    misclassification_erm_avg = 0 
    misclassification_erm_min = 1
    misclassification_erm_max = 0

    misclassification_cegqs_avg = 0 
    misclassification_cegqs_min = 1
    misclassification_cegqs_max = 0

    misclassification_erm_mmc_avg = 0 
    misclassification_erm_mmc_min = 1
    misclassification_erm_mmc_max = 0

    misclassification_cegqs_mmc_avg = 0 
    misclassification_cegqs_mmc_min = 1
    misclassification_cegqs_mmc_max = 0
    
    misclassification_cart_unbounded_avg = 0 
    misclassification_cart_unbounded_min = 1
    misclassification_cart_unbounded_max = 0
    misclassification_cart_bounded_avg = 0 
    misclassification_cart_bounded_min = 1
    misclassification_cart_bounded_max = 0

    misclassification_inferdt_depth_size_avg = 0 
    misclassification_inferdt_depth_size_min = 1
    misclassification_inferdt_depth_size_max = 0
    misclassification_inferdt_depth_avg = 0 
    misclassification_inferdt_depth_min = 1
    misclassification_inferdt_depth_max = 0

    misclassification_binoct_avg = 0 
    misclassification_binoct_min = 1
    misclassification_binoct_max = 0

    # CREATE EVALUATION SETS 
    num_samples = 0
    for set_number in range(num_eva):
        print("----------------------------")
        print(f"Creating evaluation set {set_number} ...")
        sampler = importlib.import_module(f".sampler",synthesis_benchmark_path.replace("/",".").rstrip('.'))
        samples  = evaluator.get_samples(sampler, evaluator.compute_cochran(delta, epsilon), synthesis_benchmark_path,f"evaluation_samples_{set_number}")
        num_samples = len(samples)

        # EVALUATE SYNPLICATE ERM
        print("|--Evaluating erm program ...")
        # erm_ce_count = evaluator.evaluate_synplicate_dd(samples,erm_program_path)
        # rate = erm_ce_count/num_samples
        # print(rate)
        # if rate < misclassification_erm_min:
        #     misclassification_erm_min = rate
        # if rate > misclassification_erm_max:
        #     misclassification_erm_max = rate
        # misclassification_erm_avg = (misclassification_erm_avg*set_number + rate )/(set_number+1)

        # EVALUATE SYNPLICATE ERM MMC
        if mmc_flag:
            print("|--Evaluating erm mmc program ...")
            erm_mmc_ce_count = evaluator.evaluate_synplicate_dd(samples,erm_mmc_program_path)
            rate = erm_mmc_ce_count/num_samples
            print(rate)
            if rate < misclassification_erm_mmc_min:
                misclassification_erm_mmc_min = rate
            if rate > misclassification_erm_mmc_max:
                misclassification_erm_mmc_max = rate
            misclassification_erm_mmc_avg = (misclassification_erm_mmc_avg*set_number + rate )/(set_number+1)


        # EVALUATE SYNPLICATE CEGS
        print("|--Evaluating cegs program ...")
        cegqs_ce_count = evaluator.evaluate_synplicate_dd(samples,cegqs_program_path)
        rate = cegqs_ce_count/num_samples
        print(rate)
        if rate < misclassification_cegqs_min:
            misclassification_cegqs_min = rate
        if rate > misclassification_cegqs_max:
            misclassification_cegqs_max = rate
        misclassification_cegqs_avg = (misclassification_cegqs_avg*set_number + rate )/(set_number+1)

        # EVALUATE SYNPLICATE CEGS MMC
        if mmc_flag:
            print("|--Evaluating cegs mmc program ...")
            cegqs_mmc_ce_count = evaluator.evaluate_synplicate_dd(samples,cegqs_mmc_program_path)
            rate = cegqs_mmc_ce_count/num_samples
            print(rate)
            if rate < misclassification_cegqs_mmc_min:
                misclassification_cegqs_mmc_min = rate
            if rate > misclassification_cegqs_mmc_max:
                misclassification_cegqs_mmc_max = rate
            misclassification_cegqs_mmc_avg = (misclassification_cegqs_mmc_avg*set_number + rate )/(set_number+1)

        # EVLUATE INFERDT DEPTH SIZE
        if inferdt_depth_size_depth != "N/A":
            print("|--Evaluating inferdt depth/size tree ...")
            inferdt_depth_size_ce_count = evaluator.evaluate_inferdt_tree(samples,inferdt_depth_size_path)
            rate = inferdt_depth_size_ce_count/num_samples
            print(rate)
            if rate < misclassification_inferdt_depth_size_min:
                misclassification_inferdt_depth_size_min = rate
            if rate > misclassification_inferdt_depth_size_max:
                misclassification_inferdt_depth_size_max = rate
            misclassification_inferdt_depth_size_avg = (misclassification_inferdt_depth_size_avg*set_number + rate)/(set_number+1)

        # EVLUATE INFERDT DEPTH 
        if inferdt_depth_depth !="N/A":
            print("|--Evaluating inferdt depth tree ...")
            inferdt_depth_ce_count = evaluator.evaluate_inferdt_tree(samples,inferdt_depth_path)
            rate = inferdt_depth_ce_count/num_samples
            print(rate)
            if rate < misclassification_inferdt_depth_min:
                misclassification_inferdt_depth_min = rate
            if rate > misclassification_inferdt_depth_max:
                misclassification_inferdt_depth_max = rate
            misclassification_inferdt_depth_avg = (misclassification_inferdt_depth_avg*set_number + rate)/(set_number+1)

        # EVLUATE BINOCT DEPTH 
        if binoct_path != "N/A":
            print("|--Evaluating binoct tree ...")
            binoct_ce_count = evaluator.evaluate_inferdt_tree(samples,binoct_path)
            rate = binoct_ce_count/num_samples
            print(rate)
            if rate < misclassification_binoct_min:
                misclassification_binoct_min = rate
            if rate > misclassification_binoct_max:
                misclassification_binoct_max = rate
            misclassification_binoct_avg = (misclassification_binoct_avg*set_number + rate)/(set_number+1)

        # EVALUATE CART BOUNDED 
        print("|--Evaluating cart bounded tree ...")
        cart_depth_rate = cart_tree_synthesizer.evaluate(f"{synthesis_benchmark_path}samples/evaluation_samples_{set_number}.csv",cart_depth_tree)
        rate = cart_depth_rate
        print(rate)
        if rate < misclassification_cart_bounded_min:
            misclassification_cart_bounded_min = rate
        if rate > misclassification_cart_bounded_max:
            misclassification_cart_bounded_max = rate
        misclassification_cart_bounded_avg = (misclassification_cart_bounded_avg*set_number + rate)/(set_number+1)

        # EVALUATE CART UNBOUNDED 
        print("|--Evaluating cart unbounded tree ...")
        cart_unbounded_rate = cart_tree_synthesizer.evaluate(f"{synthesis_benchmark_path}samples/evaluation_samples_{set_number}.csv",cart_unbounded_tree)
        rate = cart_unbounded_rate
        print(rate)
        if rate < misclassification_cart_unbounded_min:
            misclassification_cart_unbounded_min = rate
        if rate > misclassification_cart_unbounded_max:
            misclassification_cart_unbounded_max = rate
        misclassification_cart_unbounded_avg = (misclassification_cart_unbounded_avg*set_number + rate)/(set_number+1)


    
    # SUMMARY
    print("****************************************************")
    print("====================================================")
    print("                         ERM                        ")
    print("                       -------                      ")
    print(f"DD size: {dd_size}")
    print(f"MS avg rate: {misclassification_erm_avg}")
    print(f"MS min rate: {misclassification_erm_min}")
    print(f"MS max rate: {misclassification_erm_max}")
    print(f"MS synthesis time: {erm_synthesis_time} sec")
    print(f"MMC avg rate: {misclassification_erm_mmc_avg}")
    print(f"MMC min rate: {misclassification_erm_mmc_min}")
    print(f"MMC max rate: {misclassification_erm_mmc_max}")
    print(f"MMC synthesis time: {erm_mmc_synthesis_time} sec")
    print("====================================================")
    print("\n")
    print("====================================================")
    print("                        CeGQS                       ")
    print("                       -------                      ")
    print(f"DD size: {dd_size}")
    print(f"MS avg rate: {misclassification_cegqs_avg}")
    print(f"MS min rate: {misclassification_cegqs_min}")
    print(f"MS max rate: {misclassification_cegqs_max}")
    print(f"MS Best Index: {cegqs_best_ms}")
    print(f"MS synthesis time: {cegqs_synthesis_time} sec")
    print(f"MMC avg rate: {misclassification_cegqs_mmc_avg}")
    print(f"MMC min rate: {misclassification_cegqs_mmc_min}")
    print(f"MMC max rate: {misclassification_cegqs_mmc_max}")
    print(f"MmC Best Index: {cegqs_best_mmc}")
    print(f"MMC synthesis time: {cegqs_mmc_synthesis_time} sec")
    print("====================================================")
    print("\n")
    print("====================================================")
    print("                       InferDT                      ")
    print("                       -------                      ")
    print(f"D/S tree size: {inferdt_depth_size_nodes}, leaves: {inferdt_depth_size_leaves}")
    print(f"D/S tree depth: {inferdt_depth_size_depth}")
    print(f"D/S avg rate: {misclassification_inferdt_depth_size_avg}")
    print(f"D/S min rate: {misclassification_inferdt_depth_size_min}")
    print(f"D/S max rate: {misclassification_inferdt_depth_size_max}")
    print(f"D/S synthesis time: {inferdt_synthesis_time} sec")
    print(f"D tree size: {inferdt_depth_nodes}, leaves: {inferdt_depth_leaves}")
    print(f"D tree depth: {inferdt_depth_depth}")
    print(f"D avg rate: {misclassification_inferdt_depth_avg}")
    print(f"D min rate: {misclassification_inferdt_depth_min}")
    print(f"D max rate: {misclassification_inferdt_depth_max}")
    print(f"D synthesis time: {inferdt_depth_synthesis_time} sec")
    print("====================================================")
    print("\n")
    print("====================================================")
    print("                        BinOct                      ")
    print("                       --------                     ")
    print(f"Tree size: {binoct_nodes}, leaves: {binoct_leaves}")
    print(f"D/S tree depth: {binoct_depth}")
    print(f"D/S avg rate: {misclassification_binoct_avg}")
    print(f"D/S min rate: {misclassification_binoct_min}")
    print(f"D/S max rate: {misclassification_binoct_max}")
    print(f"D/S synthesis time: {binoct_synthesis_time} sec")
    print("====================================================")
    print("\n")
    print("====================================================")
    print("                         Cart                       ")
    print("                        ------                      ")
    print(f"B tree size: {cart_bounded_nodes}")
    print(f"B tree depth: {cart_bounded_depth}")
    print(f"B avg rate: {misclassification_cart_bounded_avg}")
    print(f"B min rate: {misclassification_cart_bounded_min}")
    print(f"B max rate: {misclassification_cart_bounded_max}")
    print(f"B synthesis time: {cart_depth_synthesis_time} sec")
    print(f"U tree size: {cart_unbounded_nodes}")
    print(f"U tree depth: {cart_unbounded_depth}")
    print(f"U avg rate: {misclassification_cart_unbounded_avg}")
    print(f"U min rate: {misclassification_cart_unbounded_min}")
    print(f"U max rate: {misclassification_cart_unbounded_max}")
    print(f"U synthesis time: {cart_unbounded_synthesis_time} sec")
    print("====================================================")

    print(f"Parameters:")
    print(f"-----------")
    print(f"ERM: delta = {delta}, epsilon = {epsilon}") 
    print(f"CeGQS: delta = {0.01}, epsilon = {0.01}")  
    print(f"Number of synthesis samples: {num_samples}")  
    print("****************************************************")







dd_size = 7
delta = 0.05
epsilon = 0.05
benchmark_name = "la"
synthesizer = ms_synthesizer
dir_name = f"{benchmark_name}_{dd_size}_{delta}_{epsilon}"
num_eva = 5
num_iter = 10

experiment(dd_size,delta,epsilon,dir_name,dd_size,num_iter,num_eva)