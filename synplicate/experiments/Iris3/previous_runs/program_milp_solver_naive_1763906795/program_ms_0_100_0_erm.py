import sys
sys.path.insert(0,"experiments/Iris3//")
import feature_defs

def execute(inputs):
	program_nodes ={}
	program_nodes["2"]= "sepal_length"
	program_nodes["1"]= "sepal_length"
	program_nodes["0"]= "petal_length"

	program_edges ={}
	program_edges[("2",3)]= "species_0"
	program_edges[("2",2)]= "species_0"
	program_edges[("2",1)]= "species_0"
	program_edges[("2",0)]= "species_0"
	program_edges[("1",3)]= "species_0"
	program_edges[("1",2)]= "species_0"
	program_edges[("1",1)]= "species_1"
	program_edges[("1",0)]= "species_0"
	program_edges[("0",3)]= "species_2"
	program_edges[("0",2)]= "species_2"
	program_edges[("0",1)]= "1"
	program_edges[("0",0)]= "2"

	features = feature_defs.retrieve_feature_defs()

	value_map = {} 
	value_map["sepal_length"] = features["sepal_length"]([("sepal_length",inputs[0]),("petal_length",inputs[1]),("petal_width",inputs[2])])
	value_map["sepal_length"] = features["sepal_length"]([("sepal_length",inputs[0]),("petal_length",inputs[1]),("petal_width",inputs[2])])
	value_map["petal_length"] = features["petal_length"]([("sepal_length",inputs[0]),("petal_length",inputs[1]),("petal_width",inputs[2])])

	flag = True
	current_node = "0"
	while flag:
		current_feature = program_nodes[current_node]
		next_node = program_edges[current_node,value_map[current_feature]]
		if next_node.isdigit():
			current_node = next_node
		else:
			current_node = next_node
			flag = False
	return current_node

