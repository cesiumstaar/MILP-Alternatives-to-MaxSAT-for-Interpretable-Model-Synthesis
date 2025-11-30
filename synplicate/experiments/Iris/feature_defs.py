# feature_defs.py for the 4-feature Iris dataset

def check_num_of_inputs(inputs):
	"""Checks that there are 4 features."""
	return len(inputs) == 4

def sepal_length(inputs):
	"""Discretizes Sepal Length into 4 bins."""
	assert(check_num_of_inputs(inputs))
	value = inputs[0][1]
	if value < 5.5:
		return 0
	if value < 6.5:
		return 1
	if value < 7.5:
		return 2
	return 3

def sepal_width(inputs):
	"""Discretizes Sepal Width into 4 bins."""
	assert(check_num_of_inputs(inputs))
	value = inputs[1][1]
	if value < 2.8:
		return 0
	if value < 3.3:
		return 1
	if value < 3.8:
		return 2
	return 3

def petal_length(inputs):
	"""Discretizes Petal Length into 4 bins."""
	assert(check_num_of_inputs(inputs))
	value = inputs[2][1]
	if value < 2.0:
		return 0
	if value < 4.0:
		return 1
	if value < 6.0:
		return 2
	return 3

def petal_width(inputs):
	"""Discretizes Petal Width into 4 bins."""
	assert(check_num_of_inputs(inputs))
	value = inputs[3][1]
	if value < 0.8:
		return 0
	if value < 1.6:
		return 1
	if value < 2.4:
		return 2
	return 3

def species(outputs):
	"""Returns the predicted class for the species label."""
	value = outputs[0][1]
	return value

def retrieve_feature_defs():
	"""Returns a dictionary mapping feature names to their definition functions."""
	feature_defs = {}
	feature_defs["sepal_length"] = sepal_length
	feature_defs["sepal_width"] = sepal_width
	feature_defs["petal_length"] = petal_length
	feature_defs["petal_width"] = petal_width
	feature_defs["species"] = species
	return feature_defs