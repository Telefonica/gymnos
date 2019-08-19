#########
Models
#########

Gymnos models are a collection of models with a common API allowing their use in a pipeline of a supervised learning system. All models inherit from :class:`gymnos.models.model.Model`.

Usage
*******
.. code-block:: python

    model = gymnos.models.load("dogs_vs_cats_cnn", input_shape=[80, 80, 1])

    train_results = model.fit(X_train, y)   # train model

    predictions = model.predict(X_test)  # get predictions for input samples
    probabilities = model.predict_proba(X_test)   # get probabilities for input samples

    test_results = model.evaluate(X_test, y_test)  # evaluate model

    model.save("saved_model")  # save model to directory
    model.restore("saved_model")  # restore model from directory

All Models
*************

.. toctree::
   :maxdepth: 2

   ./image.rst
   ./structured.rst
   ./text.rst
