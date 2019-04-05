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
Add this configuration to your experiment json file

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
