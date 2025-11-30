# synplicate
The synplicate tool implements an algorithm for synthesizing Pareto-optimal interpretations for black-box machine learning models. Given a space of interpretation and measures for accuracy and interpretability, synplicate explores the solutions space for Pareto-optimal interpretations with respect to the two given measures. 

Currently, synplicate supports synthesizing interpretations from the class of bounded decision diagrams that are defined over a custom set of predicates. The underlying synthesis algorithm is based on solving a weighted MaxSat problem, where the space of solutions is encoded as a circuit and where the accuracy and interpretability measures are encoded as weights of certain clauses in the encoding. For details on the encoding, we refer the user to our [2021 FMCAD paper](https://arxiv.org/abs/2108.07307). Similar classes of interpretations can be encoded in a similar fashion.

## Dependencies

Executing synplicate requires an installation of python 3.x and the pandas and pysat libraries. 

## Usage

### Creating an experiment

*Model and Sampler:* The model can be any executable (e.g. a decision tree, Tensorflow model, pytorch model, a python script, etc.). For each model, the user has to implement a sampler. Specifically, a sampler needs to include a method *uniform* that given the number of samples returns a dictionary with key of the form ((feature1_name,feature1_value),(feature2_name,feature2_value),(feature3_name,feature3_value), ...) mapped to values of the form [(label_name,prediction_value)]. For an example see example/AutoTaxi/sampler.py .

*Predicates and Config file:* For each experiment, one needs to provide a python implementation of the predicates over which the decision diagrams are defined. Each predicate is defined as a method Predicate_name(inputs), where inputs need to have the same type of keys as the samples map. The output of each method is the label value and should be exhaustive for all label classes. For an example see /examples/AutoTaxi/feature_defs.py .

Once the feature_def file has been implemented, one needs to define the space of interpretation. This is done by writing a dd.config file, that includes the size of the decision diagram. The names, splits, and explainability weights of the predicates; make sure this is consistent with feature_def. Names of the labels and their splits (currently we support only one label with many splits). Extend for now should always be set to false.  

*Comment on explainability weights:* To compute the normalized explainability weights, weights should be divided by the maximum possible explainability weight. This has not been implemented yet and still requires manual effort. For each experiment, the maximum explainability weight needs to be computed and edited on line 271 in synthesizer/max_sat/dd_encoder.py .

### Executing synplicate
In its most simple form, synplicate can be used by executing the following command: 
  python synplicate.py <benchmark_folder_path>
  
For custom parameters see synplicate.py -h

The tool is intended for use under linux. For Mac users, replace bc2cnf in synthesizer/max_sat/ with a OS X compatible binary for [bc2cnf](https://users.ics.aalto.fi/tjunttil/circuits/)
