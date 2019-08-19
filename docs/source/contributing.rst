.. _contributing:

#############
Contributing
#############

We provide a ``Pipfile`` with all dependencies resolved for the project (including dev dependencies). First obtain `Pipenv <https://github.com/pypa/pipenv>`_ and then install Gymnos with development dependencies:

Install gymnos with development dependencies:

.. code-block:: console

  $ pipenv install --dev

This will install complete Gymnos with development dependencies like ``pytest`` or ``sphinx``.

If you have not installed TensorFlow, install it with:

.. code-block:: console

  $ pipenv run pip3 install .[tensorflow]

Or ``tensorflow-gpu`` for GPU environments:

.. code-block:: console

  $ pipenv run pip3 install .[tensorflow-gpu]

Then, enter virtual environment:

.. code-block:: console

  $ pipenv shell

Run tests
----------

.. code-block:: console

  $ pytest

To run full suite of tests including tests that download data from external services:

.. code-block:: console

  $ pytest --runslow

Build documentation
--------------------

.. code-block:: console

  $ pipenv run sphinx:build

To watch for changes and autoreload documentation on browser:

.. code-block:: console

    $ pipenv run sphinx:watch

Style guide
------------

We follow PEP8 conventions with some minor modifications for convenience and better readability:

    - We follow **E501** standard but now the max line length is **120 characters**
    - We ignore the following code standards:

        * **E221** (Multiple spaces before operator)
        * **W504** (Line break occurred after a binary operator)

To run flake8 to check if your code sticks to the conventions, run the following command:

.. code-block:: console

    $ flake8


