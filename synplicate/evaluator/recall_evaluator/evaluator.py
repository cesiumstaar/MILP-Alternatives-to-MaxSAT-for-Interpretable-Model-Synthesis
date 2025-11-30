import importlib
import itertools
import random





# radius = 1
# threshold = 0.01
# sample_rate = 20


def evaluate(sampler,program_path,samples,radius,threshold):

    # get input names
    input_names = []
    for key in (samples.keys()):
        for name, val in key:
            input_names.append(name)
        break

    test_samples = {}


    # import program
    split_path = program_path.rsplit("/",1)
    program_file_name = split_path[1][:-3]
    program_file_dir = split_path[0]
    program = importlib.import_module(f".{program_file_name}",program_file_dir.replace("/","."))

    tp = 0
    fn = 0
    newtp = 0
    newfn = 0
    distance = 0.0
    newdistance = 0.0
    # for each sample compute neighbors and compare sampler and program on each neighbor 
    for inputs, outputs in samples.items():
        # compute inputs of sample
        input_values = []
        for input_name, input_value in inputs:
            input_values.append(input_value)
        
        if program.execute(input_values) == f"{outputs[0][0]}_{outputs[0][1]}":
            tp +=1
        else:
            fn +=1

        # compute neighborhood of input_values
        neighborhood_region = []
        for i in input_values:
            region = []
            for j in range(0-radius,radius+1):
                region.append(i+j)
            neighborhood_region.append(region)


        # print(neighborhood_region) 

        # print("|---Evaluating in neighborhood...")
        for neighbor in itertools.product(*neighborhood_region):
            # print(neighbor)
            newoutputs = sampler.predict(list(neighbor))
            key = []
            for item in range(len(neighbor)):
                key.append((input_names[item],neighbor[item]))

            if not (tuple(key) in test_samples):
                test_samples[tuple(key)] = newoutputs

                if program.execute(list(neighbor)) == f"{newoutputs[0][0]}_{newoutputs[0][1]}": # TODO this works only for classifiers 
                    newtp +=1
                else:
                    newfn +=1

            # print(f"Program returns: {program.execute(list(neighbor))}\n")
            # print(f"Sampler computed:{outputs}\n")
    # print("Uniform evalutions...")
    # evaluate over uniform sampling from whole space
    # uniform_samples = sampler.uniform(10)
    utp =0
    ufn =0
    # for inputs, outputs in uniform_samples.items():
    #     # compute inputs of sample
    #     input_values = []
    #     for input_name, input_value in inputs:
    #         input_values.append(input_value)
        
    #     if program.execute(input_values) == f"{outputs[0][0]}_{outputs[0][1]}":
    #         utp +=1
    #     else:
    #         ufn +=1

    print(f"Current true positives:{tp}, and false negatives: {fn}")
    print(f"Neighbor true positives:{newtp}, and false negatives: {newfn}")
    # print(f"Uniform true positives:{utp}, and false negatives: {ufn}")

    distance = 1- tp/(fn+tp)
    print(f"Current distance: {distance}")
    newdistance = 1- newtp/(newfn+newtp)
    print(f"Neighbor distance: {newdistance}")
    # udistance = 1- utp/(ufn+utp)
    # print(f"Uniform distance: {udistance}")

    print(abs(newdistance - distance))


    return abs(newdistance - distance) <=  threshold, test_samples




def refine(sampler,program_path, samples,radius,sample_rate):

    newsamples = {}
    # import program
    split_path = program_path.rsplit("/",1)
    program_file_name = split_path[1][:-3]
    program_file_dir = split_path[0]
    program = importlib.import_module(f".{program_file_name}",program_file_dir.replace("/","."))
    
    # get input names
    input_names = []
    for key in (samples.keys()):
        for name, val in key:
            input_names.append(name)
        break


    # for each sample compute neighbors and compare sampler and program on each neighbor 
    
    count = 0
    for inputs, outputs in samples.items():
        # compute inputs of sample
        input_values = []
        for input_name, input_value in inputs:
            input_values.append(input_value)

        # compute neighborhood of input_values
        neighborhood_region = []
        for i in input_values:
            region = []
            for j in range(0-radius,radius+1):
                region.append(i+j)
            neighborhood_region.append(region)

        # print("|--Sampling in neighborhood...")
        for neighbor in itertools.product(*neighborhood_region):
            # print(count)
            newoutputs = sampler.predict(list(neighbor))
            # if program.execute(list(neighbor)) != f"{newoutputs[0][0]}_{newoutputs[0][1]}":
                # prepare key 
            key = []
            for item in range(len(neighbor)):
                key.append((input_names[item],neighbor[item]))

            if not (tuple(key) in samples):
                # print(f"Adding new sample {key} -> {newoutputs}")
                newsamples[tuple(key)] = newoutputs
                count += 1
                if count>sample_rate:
                    return newsamples
    

    return newsamples