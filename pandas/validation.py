import math

import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

import plotly.plotly as py
import plotly.graph_objs as go

# Set Tensorflow logging verbosity
tf.logging.set_verbosity(tf.logging.ERROR)

# set pandas display options; used by the describe function
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# load the csv into a Pandas dataframe (table essentially; one with multiple series)
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

# randomize the rows of the dataframe (i.e table)
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))

def preprocess_features(california_housing_dataframe):
    """Prepares input features from California housing data set.

    Args:
        california_housing_dataframe: A Pandas DataFrame expected to contain data
        from the California housing data set.
    Returns:
        A DataFrame that contains the features to be used for the model, including
        synthetic features.
    """
    selected_features = california_housing_dataframe[[
        "latitude",
        "longitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income"
    ]]
    processed_features = selected_features.copy()
    processed_features["rooms_per_person"] = (california_housing_dataframe["total_rooms"]/california_housing_dataframe["population"])

    return processed_features

def preprocess_targets(california_housing_dataframe):
    """Prepares target features (i.e., labels) from California housing data set.

    Args:
        california_housing_dataframe: A Pandas DataFrame expected to contain data
        from the California housing data set.
    Returns:
        A DataFrame that contains the target feature.
    """
    output_targets = pd.DataFrame()
    output_targets["median_house_value"] = (california_housing_dataframe["median_house_value"]/1000.0)

    return output_targets

# create the training data set
trainingExamples = preprocess_features(california_housing_dataframe.head(12000))
trainingTargets = preprocess_targets(california_housing_dataframe.head(12000))

# create the validation data set
validationExamples = preprocess_features(california_housing_dataframe.tail(5000))
validationTargets = preprocess_targets(california_housing_dataframe.tail(5000))

colorscale = [[0, '#FAEE1C'], [0.33, '#F3558E'], [0.66, '#9C1DE7'], [1, '#FF0000']]
'''
trace1 = go.Scatter(
    x = trainingExamples.longitude,
    y = trainingExamples.latitude,
    mode='markers',
    marker=dict(
        size='16',
        color = trainingTargets.median_house_value,
        colorscale=colorscale,
        showscale=True
    )
)
'''
trace1 = go.Scatter(
    x = trainingExamples.longitude,
    y = trainingExamples.latitude,
    mode='lines',
    name='interplolation'
)

data = [trace1]
py.plot(data, filename='scatter-for-dashboard')