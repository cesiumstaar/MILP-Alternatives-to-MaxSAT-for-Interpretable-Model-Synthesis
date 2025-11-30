import sys
sys.path.insert(0,"experiments/loan_acquisition//")
import feature_defs

def execute(inputs):
	program_nodes ={}
	program_nodes["0"]= "age"

	program_edges ={}
	program_edges[("0",3)]= "approved_1"
	program_edges[("0",2)]= "approved_0"
	program_edges[("0",1)]= "approved_0"
	program_edges[("0",0)]= "approved_0"

	features = feature_defs.retrieve_feature_defs()

	value_map = {} 
	value_map["age"] = features["age"]([("age",inputs[0]),("monthly_income",inputs[1]),("dependents",inputs[2]),("credit_score",inputs[3])])

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

