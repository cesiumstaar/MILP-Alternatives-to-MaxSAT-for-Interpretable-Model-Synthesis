# feature_defs.py for the 3-feature Banknote Authentication dataset

def check_num_of_inputs(inputs):
	"""Checks that there are 3 features."""
	return len(inputs) == 3

def variance(inputs):
	"""Discretizes variance into 4 bins."""
	assert(check_num_of_inputs(inputs))
	value = inputs[0][1]
	if value < -2.0:
		return 0
	if value < 0.0:
		return 1
	if value < 2.0:
		return 2
	return 3

def skewness(inputs):
	"""Discretizes skewness into 4 bins."""
	assert(check_num_of_inputs(inputs))
	value = inputs[1][1]
	if value < -5.0:
		return 0
	if value < 0.0:
		return 1
	if value < 5.0:
		return 2
	return 3

def curtosis(inputs):
	"""Discretizes curtosis into 4 bins."""
	assert(check_num_of_inputs(inputs))
	value = inputs[2][1]
	if value < 0.0:
		return 0
	if value < 5.0:
		return 1
	if value < 10.0:
		return 2
	return 3

def CLASS(outputs):
	"""Returns the predicted class (0 for genuine, 1 for forged)."""
	value = outputs[0][1]
	return value

def retrieve_feature_defs():
	"""Returns a dictionary mapping feature names to their definition functions."""
	feature_defs = {}
	feature_defs["variance"] = variance
	feature_defs["skewness"] = skewness
	feature_defs["curtosis"] = curtosis
	feature_defs["class"] = CLASS
	return feature_defs
