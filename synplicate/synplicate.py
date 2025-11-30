import importlib
import timeit
import os
import argparse

#import procedures
from methods.erm import erm_procedure as erm
# from methods.cegqs import cegqs_procedure as cegqs

# import synthesizers
from synthesizer.max_sharp_sat import mmc_synthesizer
from synthesizer.max_sat import ms_synthesizer
from synthesizer.omt import omt_synthesizer
from evaluator.recall_evaluator import evaluator





# Parse arguments
parser = argparse.ArgumentParser(description='synplicate', usage='%(prog)s [-h] [options] model_dir_path', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

##  arguments

parser.add_argument('modelDirPath', help='path to model directory', metavar='model_path')
parser.add_argument('--initNumSamples', help='number of initial samples', type=int, default=10)
parser.add_argument('--method', help='synthesis procedure: pac, cegqs', default="pac", choices=["pac","cegqs"])
parser.add_argument('--synthesizer', help='synthesis algorithm: mmc, ms, omt', default="ms", choices=["mmc","ms","omt"])
parser.add_argument('--radius', help='neighborhood radius for neighborhood sampling', type=int, default=1)
parser.add_argument('--sampleRate', help='refinement sample rate', type=int, default=5)
parser.add_argument('--robustness', help='distance between last two interpretations', type=float, default=0.01)
parser.add_argument('--stability', help='number of iteration to determine stability', type=int,default=10)
parser.add_argument('--steps', help='number of refinement steps', type=int, default=5)
parser.add_argument('--confidence', help='delta value', type=float, default=0.05)
parser.add_argument('--error_margin', help='epsilon', type=float, default=0.03)
parser.add_argument('--threshold', help='termination threshold', type=float, default=0.05)
parser.add_argument('--termination', help='termination condition for CeGQS procedure: threshold, steps, stability', default="ttf", choices=["ttt","ttf","tft","fft","tff","ftf","fft"])
parser.add_argument('--refinement_size', help='refinement size', type=int, default=10)
parser.add_argument('--solver', help='default(RC2) or MILP', type=str, default="default")

args = parser.parse_args()


synthesis_benchmark_path = args.modelDirPath + "/"
init_num_of_samples = args.initNumSamples
method = args.method
synthesizer = args.synthesizer
radius = args.radius
rate = args.sampleRate
threshold = args.robustness
stability_factor = args.stability
step_bound = args.steps
confidence = args.confidence
error_margin = args.error_margin 
termination_threshold = args.threshold
terminiation_condition = args.termination
refinement_size = args.refinement_size
solver=args.solver

# =================================================================================================
# main loop
# =================================================================================================

# Prepare synthesizer
if synthesizer == "mmc":
    synthesizer = mmc_synthesizer
if synthesizer == "ms":
    synthesizer = ms_synthesizer
if synthesizer == "omt":
    synthesizer = omt_synthesizer


# Execute based on method
if method == "pac":
    # call erm procedure with synthesis algorithm and necessary parameters
    samples, Gamma = erm.execute(synthesis_benchmark_path,synthesizer,confidence,error_margin,"ms", solver=solver)
    print(f"Pareto-optimal solutions: \n {Gamma}")
    output = open(f"{synthesis_benchmark_path}results.csv","w")
    for path, corr, expl in Gamma:
        output.write(f"{path}  ,   {corr}   ,   {expl}\n")
    output.close

if method == "cegqs":
    # call cegqs procedure with synthesis algorithm and necessary parameters 
    # cegqs.execute(synthesis_benchmark_path,synthesizer,confidence,error_margin,init_num_of_samples, step_bound, termination_threshold, stability_factor, terminiation_condition,refinement_size)
    print("Coming soon!")
# =================================================================================================
# =================================================================================================
# =================================================================================================



