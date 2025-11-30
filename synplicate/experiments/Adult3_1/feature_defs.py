# feature_defs.py for the 3-feature Adult (Census Income) dataset

def check_num_of_inputs(inputs):
	"""Checks that there are 3 features."""
	return len(inputs) == 3

# --- Numerical Feature Definitions ---
def age(inputs):
	"""Discretizes age into 4 bins."""
	assert(check_num_of_inputs(inputs))
	value = inputs[0][1]
	if value < 25:
		return 0
	if value < 40:
		return 1
	if value < 55:
		return 2
	return 3

def hours_per_week(inputs):
	"""Discretizes hours-per-week into 4 bins."""
	assert(check_num_of_inputs(inputs))
	value = inputs[1][1]
	if value < 35:
		return 0
	if value < 40:
		return 1
	if value < 50:
		return 2
	return 3

# --- Categorical Feature Definition ---
def workclass(inputs):
	"""Maps the workclass category to an integer, grouping some classes."""
	assert(check_num_of_inputs(inputs))
	value = inputs[2][1]

	if value == 'Private':
		return 0
	elif value == 'Self-emp-not-inc':
		return 1
	elif value == 'Self-emp-inc':
		return 2
	elif value in ['State-gov', 'Local-gov', 'Federal-gov']:
		return 3  # Grouped 'Government'
	elif value in ['Without-pay', 'Never-worked']:
		return 4  # Grouped 'Unemployed'
	elif value == '?':
		return 5  # 'Unknown'
	else:
		return -1

# --- Label Definition ---
def income(outputs):
	"""Maps the income label to an integer."""
	value = outputs[0][1]
	mapping = {'<=50K': 0, '>50K': 1}
	return mapping.get(value, -1)

def retrieve_feature_defs():
	"""Returns a dictionary mapping all names to their definition functions."""
	feature_defs = {}
	feature_defs["age"] = age
	feature_defs["hours_per_week"] = hours_per_week
	feature_defs["workclass"] = workclass
	feature_defs["income"] = income
	return feature_defs