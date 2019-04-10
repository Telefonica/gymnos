###############################
Datasets
###############################

.. _datasets:

This section shows a collection of datasets currently supoorted by Gymnos


MNIST
---------------------

Classification dataset of 70,000 28x28 grayscale images of 10 handwritten digits. Please visit `mnist`_ web site for a complete description.

.. _mnist: http://yann.lecun.com/exdb/mnist/

Config example:

.. code-block:: json

    {
        "dataset": {
            "name": "mnist"
        }
    }


Boston Housing
---------------------
   
Regression dataset with 13 attributes of houses at different locations around the Boston suburbs in the late 1970s. Targets are the median values of the houses at a location (in k$).

Config example:

.. code-block:: json

    {
        "dataset": {
            "name": "boston_housing"
        }
    }


CIFAR-10
---------------------

Classification Dataset of 60,000 32x32 color images, labeled over 10 categories.

Config example:

.. code-block:: json

    {
        "dataset": {
            "name": "cifar10"
        }
    }


Dogs vs Cats
---------------------

Classification dataset of 150x150 color images of dogs and cats. Train your algorithm to predict whether the image is a dog or a cat (1 = dog, 0 = cat).
Please visit Kaggle website `dogs-vs-cats`_ for more details.

.. _dogs-vs-cats: https://www.kaggle.com/c/dogs-vs-cats

Config example:

.. code-block:: json

    {
        "dataset": {
            "name": "dogs_vs_cats"
        }
    }

Fashion MNIST
---------------------

Classification Dataset of 70,000 28x28 grayscale images of 10 fashion categories. This dataset can be used as a drop-in replacement for MNIST.


Config example:

.. code-block:: json

    {
        "dataset": {
            "name": "fashion_mnist"
        }
    }

IMDB
---------------------

Clasification Dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative).


Config example:

.. code-block:: json

    {
        "dataset": {
            "name": "imdb"
        }
    }

MTE
---------------------

Multi-Label clasification Dataset for Media Tagging Engine with descriptions of movies and tv shows from Movistar+. The goal is to predict the category it belongs e.g Sports, Music, etc ...


Config example:

.. code-block:: json

    {
        "dataset": {
            "name": "mte"
        }
    }

Tiny Imagenet
---------------------

Classification datasetof 64x64 color images of 200 classes.


Config example:

.. code-block:: json

    {
        "dataset": {
            "name": "tiny_imagenet"
        }
    }

Data Usage Test
---------------------

Dataset  of Yearly (1700-2008) data on sunspots from the National Geophysical Data Center. Used as test of data usage models. It is only a time serie for testing data usage models.


Config example:

.. code-block:: json

    {
        "dataset": {
            "name": "data_usage_test"
        }
    }

Unusual Data Usage Test
------------------------

Dataset  of Yearly (1700-2008) data on sunspots from the National Geophysical Data Center. Used as test of unusual data usage models. It is only a time serie for testing data usage models.


Config example:

.. code-block:: json

    {
        "dataset": {
            "name": "unusual_data_usage_test"
        }
    }

