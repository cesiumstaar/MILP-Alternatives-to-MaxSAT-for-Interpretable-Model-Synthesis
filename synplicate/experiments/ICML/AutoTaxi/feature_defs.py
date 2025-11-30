#Generated file 
def check_num_of_inputs(inputs):
	return len(inputs)==3

def clouds(inputs):
	assert(check_num_of_inputs(inputs))
	value = inputs[0][1]
	return value

def day_time(inputs):
	assert(check_num_of_inputs(inputs))
	value = inputs[1][1]
	if value < 56000.0:
		return 0
	if value < 72000.0:
		return 1
	if value < 90000.0:
		return 2
	return 3

def init_pos(inputs):
	assert(check_num_of_inputs(inputs))
	value = inputs[2][1]
	if value < -2.5:
		return 0
	if value < 0:
		return 1
	if value < 2.5:
		return 2
	return 3

def alert(outputs):
	value = outputs[0][1]
	return value

def retrieve_feature_defs(): 
	feature_defs = {} 
	feature_defs["clouds"] = clouds
	feature_defs["day_time"] = day_time
	feature_defs["init_pos"] = init_pos
	feature_defs["alert"] = alert
	return feature_defs 
