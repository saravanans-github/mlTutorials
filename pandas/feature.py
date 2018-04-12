import math
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset
from IPython import display
from sklearn import metrics

def preprocessFeatures(california_housing_dataframe):
    # selectedFeatures = californiaHousingDataframe[["latitude","longitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income"]]

    # # create synthetic features
    # processedFeatures = selectedFeatures.copy()
    # processedFeatures["rooms_per_person"] = processedFeatures.total_rooms/processedFeatures.population

    selected_features = california_housing_dataframe[
    ["latitude",
     "longitude",
     "housing_median_age",
     "total_rooms",
     "total_bedrooms",
     "population",
     "households",
     "median_income"]]
    processed_features = selected_features.copy()

    processed_features["housing_median_age"] /= 7
    # Create a synthetic feature.
    processed_features["rooms_per_person"] = (
    california_housing_dataframe["total_rooms"] /
    california_housing_dataframe["population"])
    return processed_features

def preprocessTargets(californiaHousingDataframe):
    targets = californiaHousingDataframe[["median_house_value"]].copy()
    targets.median_house_value /= 1000.0
    return targets

def tf_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}

    # create a batch
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(feature) for feature in input_features])

def train_model(learning_rate, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets):
    periods = 10
    steps_per_period = steps/periods

    # Create a linear tensor flow regressor object
    tf_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    tf_optimizer = tf.contrib.estimator.clip_gradients_by_norm(tf_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=construct_feature_columns(training_examples),optimizer=tf_optimizer)

    training_input_function = lambda: tf_input_fn(training_examples, training_targets["median_house_value"], batch_size)
    predict_training_input_function = lambda: tf_input_fn(training_examples, training_targets["median_house_value"], batch_size=1, shuffle=False, num_epochs=1)
    validation_input_function = lambda: tf_input_fn(validation_examples, validation_targets["median_house_value"], batch_size=1, shuffle=False, num_epochs=1)

    print "Training model..."
    print "RMSE on training data:"
    training_rmse = []
    validation_rmse = []

    for period in range(0,periods):
        # Train the model from the current state
        linear_regressor.train(input_fn=training_input_function, steps=steps_per_period)
        training_predictions=linear_regressor.predict(input_fn=predict_training_input_function)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions=linear_regressor.predict(input_fn=validation_input_function)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        training_root_mean_square_error = math.sqrt(metrics.mean_squared_error(training_predictions, training_targets))
        print " training period %02d : %0.2f" % (period, training_root_mean_square_error)
        training_rmse.append(training_root_mean_square_error)

        validation_root_mean_square_error = math.sqrt(metrics.mean_squared_error(validation_predictions, validation_targets))
        print " validation period %02d : %0.2f" % (period, validation_root_mean_square_error)
        training_rmse.append(validation_root_mean_square_error)


    print "Model training finished."

    return linear_regressor

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

californiaHousingDataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
californiaHousingDataframe = californiaHousingDataframe.reindex(np.random.permutation(californiaHousingDataframe.index))

# correlation = californiaHousingDataframe.copy()
# print correlation.corr()

processedFeatures = (preprocessFeatures(californiaHousingDataframe))[["median_income","housing_median_age"]]
processedTargets = preprocessTargets(californiaHousingDataframe)

training_examples = processedFeatures.head(12000)
validation_examples = processedFeatures.tail(5000)

training_targets = processedTargets.head(12000)
validation_targets = processedTargets.tail(5000)

train_model(learning_rate=0.15,steps=200,batch_size=5, 
            training_examples=training_examples, 
            training_targets=training_targets, 
            validation_examples=validation_examples, 
            validation_targets=validation_targets)