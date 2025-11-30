import os

def dump_samples(samples,path,file_name):

    os.system(f"mkdir -p {path}samples/")
    samples_file = open(f"{path}samples/{file_name}.csv","w")


    inputs = []
    outputs = []

    if len(samples)>0:
        inputs = list(samples.keys())[0]
        outputs = list(samples.values())[0]

        for name, val in inputs:
            samples_file.write(f"{name},")
        for name, val in outputs:
            samples_file.write(f"{name},")

        size = samples_file.tell()               # the size...
        samples_file.truncate(size-1) 

        samples_file.write("\n")
        
        for inputs, outputs in samples.items():
            for name, val in inputs:
                samples_file.write(f"{val},")
            for name, val in outputs:
                samples_file.write(f"{val},")
            size = samples_file.tell()               # the size...
            samples_file.truncate(size-1) 
            samples_file.write("\n")


    samples_file.close()
