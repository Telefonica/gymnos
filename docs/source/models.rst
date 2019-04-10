###############################
Models
###############################

.. _models:

This section shows a collection of models currently supported by Gymnos

***********************
Dogs vs Cats CNN
***********************
A Convolutional model trained for Dogs vs Cats dataset with two convolutional layers, max-pooling  and final dense layers. This model is implemented with Tensorflow.

Basic Config
---------------------
Add this configuration to your experiment json file:

.. code-block:: json

    {
        "model": {
            "name": "dogs_vs_cats_cnn"
        }
    }

***********************
Fashion MNIST
***********************
A deep neuronal network with fully connected layers trained for Fashion MNIST dataset. This model is implemented with Keras.

Basic Config
---------------------
Add this configuration to your experiment json file:

.. code-block:: json

    {
        "model": {
            "name": "fashion_mnist_nn"
        }
    }

***********************
Data Usage Holt-Winters
***********************
The Holt-Winters algorithm is a time-series forecasting method.
More recent historical data is assigned more weight in forecasting than the older results.
This version has modifications for data usage datasets.

Basic Config
---------------------
Add this configuration to your experiment json file:

.. code-block:: json

    {
        "model": {
            "name": "data_usage_holt_winters"
        }
    }

****************************
Data Usage Linear Regression
****************************
Fit a regression line to a time serie accumulated and select the points of this line as predictions.
This model is implemented with Scikit-Learn.

Basic Config
---------------------
Add this configuration to your experiment json file:

.. code-block:: json

    {
        "model": {
            "name": "data_usage_linear_regression"
        }
    }

***********************
Data Usage LSTM
***********************
LSTM model used to predict consumptions in a time serie.

Basic Config
---------------------
Add this configuration to your experiment json file:

.. code-block:: json

    {
        "model": {
            "name": "data_usage_lstm"
        }
    }

**************************************
Unusual Data Usage Weighted Thresholds
**************************************
It is an new anomaly detection algorithm based on determining if a prediction is above a typical deviation based on a weighted average. A sigma parameter is used to control the anomaly threshold.

Basic Config
---------------------
Add this configuration to your experiment json file:

.. code-block:: json

    {
        "model": {
            "name": "unusual_data_usage_weighted_thresholds"
        }
    }


