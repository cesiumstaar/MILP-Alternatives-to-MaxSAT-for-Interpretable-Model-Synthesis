import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

print(tf.version.VERSION)

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

print("Imported modules.")

print("Defining auxiliary functions...")

def plot_the_loss_curve(epochs, mse):
  """Plot a curve of loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Mean Squared Error")

  plt.plot(epochs, mse, label="Loss")
  plt.legend()
  plt.ylim([mse.min()*0.95, mse.max() * 1.03])
  plt.show()  

print("Defined the plot_the_loss_curve function.")

#Define functions to create and train a linear regression model
def create_model(my_learning_rate, my_feature_layer):
  """Create and compile a simple linear regression model."""
  model = tf.keras.models.Sequential()
  model.add(my_feature_layer)

  #   * units specifies the number of nodes in this layer.
  #   * activation specifies the activation function (ReLU).
  #   * name is just a string that can be useful when debugging.

  # First hidden layer with 20 nodes.   
  model.add(tf.keras.layers.Dense(units=20, 
                                  activation='relu', 
                                  name='Hidden1'))
  
  # Second hidden layer with 12 nodes. 
  model.add(tf.keras.layers.Dense(units=12, 
                                  activation='relu', 
                                  name='Hidden2'))
  
  # Output layer.
  model.add(tf.keras.layers.Dense(4))                              
  
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.MeanSquaredError()])

  return model

print("Defined the create_model functions.")

def train_model(model, dataset, epochs, label_name,
                batch_size=None):
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
  hist = pd.DataFrame(history.history)
  mse = hist["mean_squared_error"]

  return epochs, mse

print("Defined the train_model functions.")

train_df = pd.read_csv("california_housing_train_classifier.csv")
train_df = train_df.reindex(np.random.permutation(train_df.index)) # shuffle examples
# test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

print("Initilized training and test data.")

# Calculate the Z-scores of each column in the training set:
# train_df_mean = train_df.mean()
# train_df_std = train_df.std()
# train_df_norm = (train_df - train_df_mean)/train_df_std

# print(train_df_mean)
# print(train_df_std)
# print(train_df.head(10))
# print(train_df_norm.head(10))

# Calculate the Z-scores of each column in the test set.
# test_df_mean = test_df.mean()
# test_df_std = test_df.std()
# test_df_norm = (test_df - test_df_mean)/test_df_std

print("Normalized data values.")


# Create feature list: latitude x longitude, mean_income, population
feature_columns = []
resolution_in_Zs = 0.3  # 3/10 of a standard deviation.

# Bucketing of latitude and longitude.
# latitude = tf.feature_column.numeric_column("latitude")
# latitude_boundaries = list(np.arange(int(min(train_df_norm['latitude'])), 
#                                      int(max(train_df_norm['latitude'])), 
#                                      resolution_in_Zs))
# latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column, latitude_boundaries)

# longitude = tf.feature_column.numeric_column("longitude")
# longitude_boundaries = list(np.arange(int(min(train_df_norm['longitude'])), 
#                                       int(max(train_df_norm['longitude'])), 
#                                       resolution_in_Zs))
# longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column, 
                                                # longitude_boundaries)

# Create a feature cross of latitude and longitude.
# latitude_x_longitude = tf.feature_column.crossed_column([latitude, longitude], hash_bucket_size=100)
# crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
# feature_columns.append(latitude) 
# feature_columns.append(longitude)  

# Represent median_income as a floating-point value.
median_income = tf.feature_column.numeric_column("median_income")
feature_columns.append(median_income)

# Represent population as a floating-point value.
population = tf.feature_column.numeric_column("population")
feature_columns.append(population)

# Create feature layer 
my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

print("Created feature layer.")

# Learning!!!
print("Learning...")
# Hyperparameters.
learning_rate = 0.03
epochs = 50
batch_size = 1000
label_name = "Class"

# Establish the model's topography.
my_model = create_model(learning_rate, my_feature_layer)

# Train the model on the normalized training set. Passing the entire
# normalized training set, but the model will only use the features
# defined by the feature_layer.
epochs, mse = train_model(my_model, train_df, epochs, 
                          label_name, batch_size)
#plot_the_loss_curve(epochs, mse)

# After building a model against the training set, test that model
# against the test set.
# test_features = {name:np.array(value) for name, value in test_df_norm.items()}
# test_label = np.array(test_features.pop(label_name)) # isolate the label
print("\n Evaluate the new model against the test set:")
# my_model.evaluate(x = test_features, y = test_label, batch_size=batch_size)


# Save the model

my_model.summary()

soft_model = tf.keras.Sequential([my_model,tf.keras.layers.Softmax()])

soft_model.predict({'latitude':np.array([-114.0]),'longitude':np.array([34.0]),'population':np.array([10000]),'median_income':np.array([10000.0])})
tf.keras.models.save_model(soft_model,"model")

# test_features = {name:np.array(value) for name, value in test_df_norm.items()}
# test_label = np.array(test_features.pop(label_name)) # isolate the label

# predictions = my_model.predict(test_features)
# print(predictions)



