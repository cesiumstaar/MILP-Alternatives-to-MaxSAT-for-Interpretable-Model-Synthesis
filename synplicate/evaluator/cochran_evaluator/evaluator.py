import importlib
import itertools
import random
import math
import tqdm

from utils import logger


def refine(sampler,program_path,samples, bound, refinement_size):

    refinement = {}
    ce_count = 0 

    # import program
    split_path = program_path.rsplit("/",1)
    program_file_name = split_path[1][:-3]
    program_file_dir = split_path[0]
    program = importlib.import_module(f".{program_file_name}",program_file_dir.replace("/",".")) 

    for i in range(bound):

        # Retrieve samples 
        s = sampler.uniform(1)

        if not s.items() <= samples.items():
            # Get inputs
            inputs = []
            for name, val in  list(s.keys())[0]:
                inputs.append(val)
            # Get outputs
            outputs =  list(s.values())[0]

            if program.execute(inputs) != f"{outputs[0][0]}_{outputs[0][1]}": #  TODO this works only for classifiers
                if ce_count<refinement_size:
                    refinement.update(s)
                    ce_count +=1
                else:
                    break
        
    return refinement


def evaluate(sampler,benchmark_path,program_path,samples,delta, epsilon, refinement_size, file_name):

    misclassification_rate = 1.0
    refinement = {}
    eva_samples = {}
    z_value = 0

    if delta == 0.05:
        z_value = 1.96
    elif delta == 0.01:
        z_value = 2.33
    else:
        raise ValueError(f"z-value for delta {delta} unknown")


    # Compute number of samples using Cochran's formula 
    # TODO for now only implemented for confidence 0.05
    num_of_samples = math.ceil((z_value**2)*0.25/(epsilon**2))
    print(f"|---Using {num_of_samples} samples")



    # import program
    split_path = program_path.rsplit("/",1)
    program_file_name = split_path[1][:-3]
    program_file_dir = split_path[0]
    program = importlib.import_module(f".{program_file_name}",program_file_dir.replace("/",".")) 


    ce_count = 0

    # Sample num of samples many samples and evaluate program
    for i in tqdm.tqdm(range(num_of_samples)):
        
        # Retrieve samples 
        s = sampler.uniform(1)
        eva_samples.update(s)
        # Get inputs
        inputs = []
        for name, val in  list(s.keys())[0]:
            inputs.append(val)
        # Get outputs
        outputs =  list(s.values())[0]
        
        try:
            if program.execute(inputs) != f"{outputs[0][0]}_{outputs[0][1]}": #  TODO this works only for classifiers
                ce_count +=1
                if ce_count<refinement_size:
                    refinement.update(s)
        except:
            ce_count = 0
            break
                
        
    
    misclassification_rate = ce_count/num_of_samples

    logger.dump_samples(eva_samples,benchmark_path,file_name)

    return refinement, misclassification_rate


def just_evaluate(sampler,benchmark_path,erm_program_path,erm_mmc_program_path, cegqs_program_path, cegqs_mmc_program_path,delta, epsilon, file_name):

    
    eva_samples = {}

    z_value = 0

    if delta == 0.05:
        z_value = 1.96
    elif delta == 0.01:
        z_value = 2.33
    else:
        raise ValueError(f"z-value for delta {delta} unknown")

    # Compute number of samples using Cochran's formula 
    # TODO for now only implemented for confidence 0.05
    num_of_samples = math.ceil((z_value**2)*0.25/(epsilon**2))
    print(f"|---Using {num_of_samples} samples")



    # import erm program
    split_path = erm_program_path.rsplit("/",1)
    program_file_name = split_path[1][:-3]
    program_file_dir = split_path[0]
    erm_program = importlib.import_module(f".{program_file_name}",program_file_dir.replace("/",".")) 

    split_path = erm_mmc_program_path.rsplit("/",1)
    program_file_name = split_path[1][:-3]
    program_file_dir = split_path[0]
    erm_mmc_program = importlib.import_module(f".{program_file_name}",program_file_dir.replace("/",".")) 



    # import cegqs program
    split_path = cegqs_program_path.rsplit("/",1)
    program_file_name = split_path[1][:-3]
    program_file_dir = split_path[0]
    cegqs_program = importlib.import_module(f".{program_file_name}",program_file_dir.replace("/",".")) 

    split_path = cegqs_mmc_program_path.rsplit("/",1)
    program_file_name = split_path[1][:-3]
    program_file_dir = split_path[0]
    cegqs_mmc_program = importlib.import_module(f".{program_file_name}",program_file_dir.replace("/",".")) 


    erm_ce_count = 0
    cegqs_ce_count = 0
    erm_mmc_ce_count = 0
    cegqs_mmc_ce_count = 0

    dont_enter_again = False
    dont_enter_again_cegqs = False

    # Sample num of samples many samples and evaluate program
    for i in tqdm.tqdm(range(num_of_samples)):
        
        # Retrieve samples 
        s = sampler.uniform(1)
        eva_samples.update(s)
        # Get inputs
        inputs = []
        for name, val in  list(s.keys())[0]:
            inputs.append(val)
        # Get outputs
        outputs =  list(s.values())[0]
        
        if erm_program.execute(inputs) != f"{outputs[0][0]}_{outputs[0][1]}": #  TODO this works only for classifiers
            erm_ce_count +=1

        if cegqs_program.execute(inputs) != f"{outputs[0][0]}_{outputs[0][1]}": #  TODO this works only for classifiers
            cegqs_ce_count +=1    

        try:
            if not dont_enter_again:
                if erm_mmc_program.execute(inputs) != f"{outputs[0][0]}_{outputs[0][1]}": #  TODO this works only for classifiers
                    erm_mmc_ce_count +=1
        except:
            dont_enter_again = True
            erm_mmc_ce_count =  0

        try:
            if not dont_enter_again_cegqs:
                if cegqs_mmc_program.execute(inputs) != f"{outputs[0][0]}_{outputs[0][1]}": #  TODO this works only for classifiers
                    cegqs_mmc_ce_count +=1    
        except:
            dont_enter_again_cegqs = True
            cegqs_mmc_ce_count = 0
                        
        
    
    erm_misclassification_rate = erm_ce_count/num_of_samples
    cegqs_misclassification_rate = cegqs_ce_count/num_of_samples
    erm_mmc_misclassification_rate = erm_mmc_ce_count/num_of_samples
    cegqs_mmc_misclassification_rate = cegqs_mmc_ce_count/num_of_samples

    logger.dump_samples(eva_samples,benchmark_path,file_name)

    return  erm_misclassification_rate,  erm_mmc_misclassification_rate, cegqs_misclassification_rate, cegqs_mmc_misclassification_rate

    
def compute_cochran(delta, epsilon):
    z_value = 0

    if delta == 0.05:
        z_value = 1.96
    elif delta == 0.01:
        z_value = 2.33
    else:
        raise ValueError(f"z-value for delta {delta} unknown")

    # Compute number of samples using Cochran's formula 
    num_of_samples = math.ceil((z_value**2)*0.25/(epsilon**2))

    return num_of_samples


def get_samples(sampler,num_of_samples,output_path,file_name):
        
    samples = {}
    for i in tqdm.tqdm(range(num_of_samples)):
    
        # Retrieve samples 
        s = sampler.uniform(1)
        samples.update(s)

    logger.dump_samples(samples,output_path,file_name)
    
    return samples

def evaluate_synplicate_dd(samples, program_path):

    ce_count = 0
    # IMPORT PROGRAM
    split_path = program_path.rsplit("/",1)
    program_file_name = split_path[1][:-3]
    program_file_dir = split_path[0]
    program = importlib.import_module(f".{program_file_name}",program_file_dir.replace("/",".")) 

    for key, value in tqdm.tqdm(samples.items()):
        # Get inputs
        inputs = []
        for in_name, in_val in key:
            inputs.append(in_val)
        # Get outputs
        outputs =  list(value)

        if program.execute(inputs) != f"{outputs[0][0]}_{outputs[0][1]}": #  TODO this works only for classifiers
            ce_count += 1

    return ce_count

def evaluate_inferdt_tree(samples, program_path):
    
    ce_count = 0
    # IMPORT PROGRAM
    split_path = program_path.rsplit("/",1)
    program_file_name = split_path[1][:-3]
    program_file_dir = split_path[0]
    program = importlib.import_module(f".{program_file_name}",program_file_dir.replace("/",".")) 

    for key, value in tqdm.tqdm(samples.items()):
        # Get inputs
        inputs = {}
        for in_name, in_val in key:
            inputs[in_name]=in_val
        # Get outputs
        outputs =  list(value)

        if program.execute(inputs) != outputs[0][1]: #  TODO this works only for classifiers
            ce_count += 1

    return ce_count