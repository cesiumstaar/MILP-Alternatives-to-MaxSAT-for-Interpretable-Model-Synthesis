import sys
sys.path.insert(0,"experiments/theorem_prover//")
import feature_defs

def execute(inputs):
	program_nodes ={}
	program_nodes["3"]= "F10"
	program_nodes["0"]= "F1"

	program_edges ={}
	program_edges[("3",2)]= "H1_1"
	program_edges[("3",1)]= "H1_1"
	program_edges[("3",0)]= "H1_1"
	program_edges[("0",3)]= "H1_1"
	program_edges[("0",2)]= "H1_1"
	program_edges[("0",1)]= "3"
	program_edges[("0",0)]= "H1_0"

	features = feature_defs.retrieve_feature_defs()

	value_map = {} 
	value_map["F10"] = features["F10"]([("F1",inputs[0]),("F10",inputs[1])])
	value_map["F1"] = features["F1"]([("F1",inputs[0]),("F10",inputs[1])])

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

