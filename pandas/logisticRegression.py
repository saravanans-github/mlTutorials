import math
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

californiaHousingDataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
californiaHousingDataframe = californiaHousingDataframe.reindex(np.random.permutation(californiaHousingDataframe.index))

def get_quantile_based_boundary(feature_values):
    boundary = feature_values.quantile(0.75)
    return boundary

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

    # Create a synthetic feature.
    processed_features["rooms_per_person"] = (
    california_housing_dataframe["total_rooms"] /
    california_housing_dataframe["population"])

    return processed_features

def preprocessTargets(californiaHousingDataframe):
    targets = pd.DataFrame()
    boundary = get_quantile_based_boundary(californiaHousingDataframe.median_house_value)
    targets["median_house_value_is_high"] = (californiaHousingDataframe.median_house_value > boundary).astype(float)

    return targets

def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

    Returns:
        A set of feature columns
    """

    return set([tf.feature_column.numeric_column(feature) for feature in input_features])

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

def train_model(learning_rate, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets):
    periods = 10
    steps_per_period = steps/periods

    # Create a linear tensor flow regressor object
    tf_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    tf_optimizer = tf.contrib.estimator.clip_gradients_by_norm(tf_optimizer, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(feature_columns=construct_feature_columns(training_examples),optimizer=tf_optimizer)

    training_input_function = lambda: tf_input_fn(training_examples, training_targets["median_house_value_is_high"], batch_size)
    predict_training_input_function = lambda: tf_input_fn(training_examples, training_targets["median_house_value_is_high"], batch_size=1, shuffle=False, num_epochs=1)
    validation_input_function = lambda: tf_input_fn(validation_examples, validation_targets["median_house_value_is_high"], batch_size=1, shuffle=False, num_epochs=1)

    print "Training model..."
    print "LogLoss on training data:"
    training_log_loss = []
    validation_log_loss = []

    for period in range(0,periods):
        # Train the model from the current state
        linear_classifier.train(input_fn=training_input_function, steps=steps_per_period)
        training_predictions=linear_classifier.predict(input_fn=predict_training_input_function)
        training_predictions = np.array([item['probabilities'] for item in training_predictions])

        validation_predictions=linear_classifier.predict(input_fn=validation_input_function)
        validation_predictions = np.array([item['probabilities'] for item in validation_predictions])

        training_ll = metrics.log_loss(training_targets, training_predictions)
        print " training period %02d : %0.2f" % (period, training_ll)
        training_log_loss.append(training_ll)

        validation_ll = metrics.log_loss(validation_targets, validation_predictions)
        print " validation period %02d : %0.2f" % (period, validation_ll)
        validation_log_loss.append(validation_ll)

    print "Model training finished."

    evaluation_metrics = linear_classifier.evaluate(input_fn=validation_input_function)

    print "AUC on the validation set: %0.2f" % evaluation_metrics['auc']
    print "Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy']

    return linear_classifier

processedFeatures = preprocessFeatures(californiaHousingDataframe)
processedTargets = preprocessTargets(californiaHousingDataframe)

training_examples = processedFeatures.head(12000)
validation_examples = processedFeatures.tail(5000)

training_targets = processedTargets.head(12000)
validation_targets = processedTargets.tail(5000)

linear_classifier = train_model(
    learning_rate=0.000003,
    steps=20000,
    batch_size=500,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)