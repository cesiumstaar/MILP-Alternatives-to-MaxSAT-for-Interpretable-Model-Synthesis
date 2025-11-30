import sys
sys.path.insert(0,"experiments/Banknote3//")
import feature_defs

def execute(inputs):
	program_nodes ={}
	program_nodes["2"]= "variance"
	program_nodes["1"]= "curtosis"
	program_nodes["0"]= "skewness"

	program_edges ={}
	program_edges[("2",3)]= "class_0"
	program_edges[("2",2)]= "class_0"
	program_edges[("2",1)]= "class_1"
	program_edges[("2",0)]= "class_1"
	program_edges[("1",3)]= "class_0"
	program_edges[("1",2)]= "class_0"
	program_edges[("1",1)]= "2"
	program_edges[("1",0)]= "2"
	program_edges[("0",3)]= "class_0"
	program_edges[("0",2)]= "1"
	program_edges[("0",1)]= "1"
	program_edges[("0",0)]= "2"

	features = feature_defs.retrieve_feature_defs()

	value_map = {} 
	value_map["variance"] = features["variance"]([("variance",inputs[0]),("skewness",inputs[1]),("curtosis",inputs[2])])
	value_map["curtosis"] = features["curtosis"]([("variance",inputs[0]),("skewness",inputs[1]),("curtosis",inputs[2])])
	value_map["skewness"] = features["skewness"]([("variance",inputs[0]),("skewness",inputs[1]),("curtosis",inputs[2])])

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

