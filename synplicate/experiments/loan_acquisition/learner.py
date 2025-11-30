import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

print(tf.version.VERSION)

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

print("Imported modules.")

print("Defining auxiliary functions...")



#Define functions to create and train a linear regression model
def create_model(my_learning_rate, my_feature_layer):
  """Create and compile a simple linear regression model."""
  model = tf.keras.models.Sequential()
  model.add(my_feature_layer)

  #   * units specifies the number of nodes in this layer.
  #   * activation specifies the activation function (ReLU).
  #   * name is just a string that can be useful when debugging.

  # First hidden layer with 20 nodes.   
  model.add(tf.keras.layers.Dense(units=200, 
                                  activation='relu', 
                                  name='Hidden1'))
  
  # # Second hidden layer with 12 nodes. 
  model.add(tf.keras.layers.Dense(units=200, 
                                  activation='relu', 
                                  name='Hidden2'))

  # # # third hidden layer with 12 nodes. 
  model.add(tf.keras.layers.Dense(units=200, 
                                  activation='relu', 
                                  name='Hidden3'))
  
  # Output layer.
  model.add(tf.keras.layers.Dense(2)) 

  # model.add(tf.keras.layers.Softmax())
  
  # model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
  #               loss="mean_squared_error",
  #               metrics=[tf.keras.metrics.MeanSquaredError()])

  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

  return model

def train_model(model, dataset, epochs, label_name,batch_size=None):
  """Train the model by feeding it data."""

  # Split the dataset into features and label.
  features = {name:np.array(value) for name, value in dataset.items()}
  label = np.array(features.pop(label_name))
  history = model.fit(x=features, y=label, batch_size=batch_size,
                      epochs=epochs, shuffle=True) 

  # The list of epochs is stored separately from the rest of history.
  epochs = history.epoch
  
  # To track the progression of training, gather a snapshot
  # of the model's mean squared error at each epoch. 
  # hist = pd.DataFrame(history.history)
  # mse = hist["mean_squared_error"]

  return epochs

train_df = pd.read_csv("loan_acquisition_age_income_bias.csv")
train_df = train_df.reindex(np.random.permutation(train_df.index)) # shuffle examples

# Create feature list: latitude x longitude, mean_income, population
feature_columns = []

age = tf.feature_column.numeric_column("age")
feature_columns.append(age)

credit_score = tf.feature_column.numeric_column("credit_score")
feature_columns.append(credit_score)

dependents = tf.feature_column.numeric_column("dependents")
feature_columns.append(dependents)

monthly_income = tf.feature_column.numeric_column("monthly_income")
feature_columns.append(monthly_income)

# Create feature layer 
my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# Learning!!!
print("Learning...")
# Hyperparameters.
learning_rate = 0.01
epochs = 100
batch_size = 1000
label_name = "approved"

# Establish the model's topography.
my_model = create_model(learning_rate, my_feature_layer)

# Train the model on the normalized training set. Passing the entire
# normalized training set, but the model will only use the features
# defined by the feature_layer.
epochs  = train_model(my_model, train_df, epochs, label_name, batch_size)

soft_model = tf.keras.Sequential([my_model,tf.keras.layers.Softmax()])

soft_model.predict({'age':np.array([20]),'credit_score':np.array([303]),'dependents':np.array([2]),'monthly_income':np.array([2000])})

# epochs  = train_model(my_model, train_df, epochs, label_name, batch_size)

tf.keras.models.save_model(soft_model,"model")
my_model.summary()