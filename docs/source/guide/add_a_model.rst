####################
Add a model
####################

Overview
============
Machine Learning models are distributed in all kinds of API specifications and in all kinds of places, and they're not always implemented in a format that's ready to feed into a machine learning pipeline. Enter Gymnos models.
Gymnos models provides a way to transform all those models into a standard format to make them ready for a machine learning pipeline.

To enable this, each model implements a subclass of :class:`gymnos.models.Model`, which specifies:

* How to train model
* How to predict samples
* How to save and restore model

Writing ``my_model.py``
==========================

Use the default template
---------------------------
If you want to :ref:`contribute to our repo <contributing>` and add a new model, the following script will help you get started generating the required python files. To use it, clone the `Gymnos <https://github.com/Telefonica/gymnos>`_ repository and run the following command:

.. code-block:: console

    $ python3 -m scripts.create_new model --name my_model

This command will create ``gymnos/models/my_model.py``, and modify ``gymnos/__init__.py`` to register model so we can load it using ``gymnos.load``.

The model registration process is done by associating the model name with their path:

.. code-block:: python
    :caption: gymnos/__init__.py

    models.register(
        name="my_model",
        entry_point="gymnos.models.my_model.MyModel"
    )

Go to ``gymnos/models/my_model.py`` and then search for TODO(my_model) in the generated file to do the modifications.

Model
-------
Each model is defined as a subclass of :class:`gymnos.models.Model` implementing the following methods:

* ``fit``: fits model to training data
* ``predict``: Generates output predictions for the input samples.
* ``evaluate``: Evaluate model
* ``save``: Save model to restore it later
* ``restore``: Restore model from checkpoint

Because of the different nature of each ML model, the following methods are not mandatory and you should only implement them if your model allows it:

* ``fit_generator``: fits the model on data generated batch-by-batch by a Python sequence / generator. Only if your model supports online-learning.
* ``predict_proba``: Generates output probabilities for the input samples (for classification tasks). Only if your model is probabilistic.

my_model.py
------------

``my_model.py`` first look like this:

.. code-block:: python

    #
    #
    #   MyModel
    #
    #

    from .model import Model

    class MyModel(Model):
        """
        TODO(my_model): Description of my model.
        """

        def __init__(self, **parameters):
            # TODO(my_model): Define and initialize model parameters.

        def fit(self, X, y, **training_parameters):
            # TODO(my_model): Fit model to training data.

        def fit_generator(self, X, y, **training_parameters):
            # OPTIONAL: Fit model to training generator. Write method if your model supports incremental learning
            raise NotImplementedError()

        def predict(self, X):
            # TODO(my_model): Predict classes/values using features.

        def predict_proba(self, X):
            # OPTIONAL: Predict probabilities using features. Write method if your model is a probabilistic model
            raise NotImplementedError()

        def save(self, save_dir):
            # TODO(my_model): Save model to save_dir.

        def restore(self, save_dir):
            # TODO(my_model): Restore model from save_dir.

Specifying ``parameters``
===========================

Use the constructor to specify any parameters you need to build your model. These parameters may be required or optional although optional parameters are preferable.

.. code-block:: python

    class MyModel(Model):

        def __init__(self, eta=0.5, penalty="l2", learning_rate=0.01):
            self.eta = eta
            self.penalty = penalty
            self.learning_rate = learning_rate

Training model
===========================

Fit model to training data specifying any parameters you need to train your model. Optional parameters are preferable.

It returns a dictionnary with training metrics.

.. code-block:: python

    def fit(self, X, y, batch_size=32, epochs=10):
        ...

        return {
            "accuracy": accuracy_epochs
        }

If your model supports online-learning by fitting a sequence, you can also write ``fit_generator`` method:

.. code-block:: python

    def fit_generator(self, generator, epochs=10):
        ... # generator will be a sequence that returns a tuple (X_batch, y_batch)
        return {
            "accuracy": accuracy_epochs
        }

Predicting input samples
===========================

Implement this method to predict values from input samples. By convention, for classification tasks, it must return the class index (e.g 2).

It returns a NumPy array with predictions.

.. code-block:: python

    def predict(self, X):
        ...

        return predictions

If your model is probabilistic, you can also write ``predict_proba`` method to return class probabilities:

.. code-block:: python

    def predict_proba(self, X):
        ...
        return probabilities

Evaluating performance
===========================

Returns metrics values for the model in test mode.

It returns a dictionnary with testing metrics

.. code-block:: python

    def evaluate(self, X, y):
        ...
        return {
            "recall": recall
        }


Saving and restoring
===========================

Save trained model.

.. code-block:: python

    def save(self, save_dir):
        self.model.save(os.path.join(save_dir, "session.pkl"))
        self.model.save_weights(os.path.join(save_dir, "weights.h5"))


Restore trained model

.. code-block:: python

    def restore(self, save_dir):
        self.model.load(os.path.join(save_dir, "session.pkl"))
        self.model.load_weights(os.path.join(save_dir, "weights.h5"))

Summary
=============

1. Create ``MyModel`` in ``gymnos/model/my_model.py`` inheriting from :class:`gymnos.models.model.Model` and implementing the abstract methods:

* ``fit``
* ``fit_generator`` (optional)
* ``predict``
* ``evaluate``
* ``save``
* ``restore``

2. Register the model in ``gymnos/__init__.py`` by adding:

.. code-block:: python

    models.register(
        name="my_model",
        entry_point="gymnos.models.my_model.MyModel"
    )

Don't Repeat Yourself with mixins
===================================

Si creamos dos modelos diferentes con el mismo framework, como puede ser Keras o Scikit-Learn, es muy probable que muchos de los métodos abstractos que hemos de sobreescribir sean identicos y cada vez que quisieramos crear un modelo con esa librería deberíamos volver a escribir esos mismos métodos. 

Veamos un ejemplo:

.. code-block:: python

    class Model1(Model):

        def __init__(self, C=0.1):
            self.model = sklearn.svm.SVC(C=C)

        def fit(self, X, y):
            self.model.fit(X)
            return {}

        def predict(self, X):
            return self.model.predict(X)

        def evaluate(self, X, y):
            return {
                "accuracy": sklearn.metrics.accuracy(self.model.predict(X), y)
            }

        ...

    class Model2(Model):

        def __init__(self, n_estimators=20):
            self.model = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators)

        def fit(self, X, y):
            self.model.fit(X, y)
            return {}

        def predict(self, X):
            return self.model.predict(X)

        def evaluate(self, X, y):
            return {
                "accuracy": sklearn.metrics.accuracy(self.model.predict(X), y)
            }

        ...

Observamos que existe mucha repetición de código entre los 2 modelos, los métodos ``fit``, ``predict``, ``evaluate`` son identicos en ambos modelos. Esto rompe con la filosofía DRY (Don't Repeat Yourself). Es por eso que hemos creado mixins para las siguientes librerías:

- Keras
- Scikit-Lern
- TensorFlow

Keras mixin
-------------

Keras classifier mixin
^^^^^^^^^^^^^^^^^^^^^^^^

Utiliza este mixin si estás creando un clasificador utilizando un modelo de Keras. Provee de los siguientes métodos:

- ``fit``
- ``fit_generator``
- ``predict``
- ``predict_proba``
- ``evaluate``
- ``save``
- ``restore``

Hereda de ``KerasClassifierMixin`` y asigna a la variable ``self.model`` tu modelo de Keras. De esa forma ya tendrás todos los métodos implementados:

.. code-block:: python

    from mixins imports KerasClassifierMixin

    class MyModel(KerasClassifierMixin, Model):

        def __init__(self, ...):
            self.model = keras.models.Model(
                ...
            )

Keras regressor mixin
^^^^^^^^^^^^^^^^^^^^^^^^

Utiliza este mixin si estás creando un regresor utilizando un modelo de Keras. Provee de los siguientes métodos:

- ``fit``
- ``fit_generator``
- ``predict``
- ``evaluate``
- ``save``
- ``restore``

Hereda de ``KerasRegressorMixin`` y asigna a la variable ``self.model`` tu modelo de Keras. De esa forma ya tendrás todos los métodos implementados:

.. code-block:: python

    from mixins imports KerasRegressorMixin

    class MyModel(KerasRegressorMixin, Model):

        def __init__(self, ...):
            self.model = keras.models.Model(
                ...
            )

Scikit-Learn mixin
-------------------

Utiliza este mixin si estás utilizando un estimador con la librería Scikit-Learn. Provee de los siguientes métodos:

- ``fit``
- ``fit_generator``: solo si el estimador tiene implementado el método ``partial_fit``.
- ``predict``
- ``predict_proba``: solo si el estimador tiene implementado el método ``predict_proba``
- ``evaluate``
- ``save``
- ``restore``

Hereda de ``SklearnMixin`` y asigna a la variable ``self.model`` tu estimador de Scikit-Learn. De esta forma ya tendrás todos los métodos implementados:

.. code-block:: python

    from mixins import SklearnMixin

    class MyModel(SklearnMixin, Model):

        def __init__(self, ...):
            self.model = sklearn.svm.SVC(
                ...
            )

TensorFlow mixin
---------------------------

Utiliza este mixin si estás utilizando una session de TensorFlow. Provee de los siguientes métodos:

- ``save``
- ``restore``

Hereda de ``TensorFlowSaverMixin`` y asigna a la variable ``self.sess`` tu sesión de TensorFlow. De esta forma ya tendrás esos métodos implementados:

.. code-block:: python

    from mixins import TensorFlowSaverMixin

    class MyModel(TensorFlowSaverMixin, Model):

        def __init__(self, ...):
            ...
            self.sess = tf.Session()


Adding the model to ``Telefonica/gymnos``
==========================================

If you'd like to share your work with the community, you can check in your model implementation to Telefonica/gymnos. Thanks for thinking of contributing!

Before you send your pull request, follow these last few steps (check :ref:`contributing` to see more details):

1. Test model with any Gymnos dataset
----------------------------------------
Check that your model is working with a Gymnos dataset.

2. Add documentation
----------------------
Add model documentation.

3. Check your code style
--------------------------
Follow the `PEP8 Python style guide <https://www.python.org/dev/peps/pep-0008/>`_, except Gymnos uses 120 characters as maximum line length.

You can lint files running ``flake8`` command:

.. code-block:: console

    $ flake8

Adding the model from other repository
=================================================

You can also add a model from other repository in a very simple way by converting your repository in a Python library.

Once you have defined your ``setup.py``, create and register your Gymnos models in the same way we have shown.

Here is a minimal example. Say we have our library named ``gymnos_my_models`` and we want to add the model ``my_model``. You have to:

1. Create ``MyModel`` in ``gymnos_my_models/my_model.py`` inheriting from :class:`gymnos.models.model.Model` and implementing the abstract methods
2. Register model in your module ``__init__.py`` referencing the name and the path:

.. code-block:: python
    :caption: gymnos_my_models/__init__.py

    import gymnos

    gymnos.models.register(
        name="my_model",
        entry_point="gymnos_my_models.my_model.MyModel"
    )


That's it, when someone wants to run ``my_model`` from ``gymnos_my_models``, simply ``pip install`` the package and reference the package when you are loading the model with the following format: ``<module_name>:<model_name>``.

For example:

.. code-block:: python

    gymnos.models.load("gymnos_my_models:my_model")
