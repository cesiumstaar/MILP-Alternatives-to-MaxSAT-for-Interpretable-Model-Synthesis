import importlib
import math
import os

# utils
from utils import logger
import timeit
# from synthesizer.decision_trees import cart_tree_synthesizer



def execute(benchmark_path, synthesizer, delta, epsilon,name,samples={},solver="default"):
    print(f"Executing PAC procedure for ({delta,epsilon})...")
    # determine number of samples based on the values of delta and epsilon 
    synthesizer.initialize(benchmark_path)

    # =================================================================================================
    # Sampler:
    # =================================================================================================
    # import sampler
    sampler = importlib.import_module(f".sampler",benchmark_path.replace("/",".").rstrip('.'))

    # Compute number of samples according to emprirical risk minimization 
    class_size = synthesizer.compute_class_size()
    num_of_samples = math.ceil(2*math.log((1+2*class_size)/delta)/(epsilon*epsilon))
    # num_of_samples = 1
    print(f"Using {num_of_samples} samples ...")
    
    if samples == {}:
        samples.update(sampler.uniform(num_of_samples))
    print(f"Got {len(samples)} samples ...")
    
    print("--- Head of Samples (first 5) ---")
    for i, (key, value) in enumerate(list(samples.items())[:5]):
        print(f"Sample {i+1}: Features: {key} -> Label: {value}")
    print("---------------------------------")
    # ----------------------------------------
    
    # =================================================================================================
    # =================================================================================================


    # =================================================================================================
    # Synthesizer:
    # =================================================================================================
    # Execute synthesizer

    
    

    # dump input samples
    logger.dump_samples(samples,benchmark_path,f"erm_syn_samples")
    samples_file = f"{benchmark_path}samples/erm_syn_samples.csv"

    # USING SKLEARN CART
    #print("Executing CART unbounded depth ...")
    ## UNBOUNDED DEPTH 
    # start = timeit.default_timer()
    # cart_unbounded_tree = cart_tree_synthesizer.synthesize(samples_file,["0","1","2","3"], benchmark_path,-1,"unbounded")
    # cart_unbounded_depth = cart_unbounded_tree.get_depth()
    # cart_unbounded_nodes = cart_unbounded_tree.get_n_leaves()
    # stop = timeit.default_timer()
    # cart_unbounded_synthesis_time = stop-start

    # initialize
    os.system(f"rm -r {benchmark_path}program")
    Gamma = []
    W = [(0,100,0)]

    while W != []:
        expl_lower_bound, expl_upper_bound, corr_bound = W.pop(0)
        synthesis_time = 0
        precision = 7
        
        # Explore in current_region
        start = timeit.default_timer()
        print("=================================")
        print(f" Exploring solution in region [{expl_lower_bound},{expl_upper_bound},{corr_bound}]")
        print("=================================")
        if (solver!="default"): program_path, dot_path, corr, explainability = synthesizer.synthesizeWithMILP(benchmark_path,samples,expl_lower_bound,expl_upper_bound,precision,f"{name}_{expl_lower_bound}_{expl_upper_bound}_{corr_bound}_erm", solver)
        else : program_path, dot_path, corr, explainability = synthesizer.synthesize(benchmark_path,samples,expl_lower_bound,expl_upper_bound,precision,f"{name}_{expl_lower_bound}_{expl_upper_bound}_{corr_bound}_erm")
        print(f"Synthesized program: {program_path}\nVisualization: {dot_path}")
        stop = timeit.default_timer()
        synthesis_time = stop-start
        print(f"Synthesis time:{synthesis_time}")

        if corr>0 and explainability>0:
            print(f"Accuracy:{corr} ({int(corr* len(samples))}/{len(samples)})")
            print(f"Explainability: {explainability}")

            if corr> corr_bound:
                Gamma.append((program_path,corr,explainability))
                print(f"Adding region ({expl_lower_bound},{int(explainability*100)},{corr})")
                W.append((expl_lower_bound,int(explainability*100),corr))
                print(f"Adding region ({int(explainability*100)},{expl_upper_bound},{corr_bound})")
                W.append((int(explainability*100),expl_upper_bound,corr_bound))
            else:
                print(f"Not Pareto-optimal. Adding only region ({expl_lower_bound},{int(explainability*100)},{corr_bound})")
                W.append((expl_lower_bound,int(explainability*100),corr_bound))
        
        print("=================================")
        W=[] ##comment this 
        # print("forcing W=[] for termination in methods/erm/erm_procedure.py")#comment this line for nomal functioning
        

    return samples, Gamma
        