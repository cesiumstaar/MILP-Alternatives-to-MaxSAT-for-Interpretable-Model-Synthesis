import sys
sys.path.insert(0,"experiments/Adult3_1//")
import feature_defs

def execute(inputs):
	program_nodes ={}
	program_nodes["1"]= "hours_per_week"
	program_nodes["0"]= "age"

	program_edges ={}
	program_edges[("1",3)]= "income_1"
	program_edges[("1",2)]= "income_0"
	program_edges[("1",1)]= "income_1"
	program_edges[("1",0)]= "income_1"
	program_edges[("0",3)]= "1"
	program_edges[("0",2)]= "1"
	program_edges[("0",1)]= "income_0"
	program_edges[("0",0)]= "income_0"

	features = feature_defs.retrieve_feature_defs()

	value_map = {} 
	value_map["hours_per_week"] = features["hours_per_week"]([("age",inputs[0]),("hours_per_week",inputs[1]),("workclass",inputs[2])])
	value_map["age"] = features["age"]([("age",inputs[0]),("hours_per_week",inputs[1]),("workclass",inputs[2])])

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

