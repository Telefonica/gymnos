########################################
Add an execution environment
########################################

Overview
============
Gymnos execution environment provides a way to train experiments in an external environment, for example, in 4th platform.

To enable this, each execution environment implements a subclass of :class:`gymnos.execution_environments.execution_environment.ExecutionEnvironment`, which specifies:

* How to train experiment
* How to monitor training status (e.g fetching logs)

Writing ``my_execution_environment.py``
=========================================

Use the default template
--------------------------
If you want to :ref:`contribute to our repo <contributing>` and add a new execution environment, the following script will help you get started generating the required python files. To use it, clone the `Gymnos <https://github.com/Telefonica/gymnos>`_ repository and run the following command:

.. code-block:: console

    $ python3 -m scripts.create_new execution_environment --name my_execution_environment

This command will create ``gymnos/execution_environments/my_execution_environment.py`` and modify ``gymnos/__init__.py`` to register execution environment so we can load it using ``gymnos.load``.

The execution environment registration process is done by associating the execution environement id with their path:

.. code-block:: python
    :caption: gymnos/__init__.py

    execution_environments.register(
        type="my_execution_environment",
        entry_point="gymnos.execution_environments.my_execution_environment.MyExecutionEnvironment"
    )

Go to ``gymnos/execution_environments/my_execution_environment.py`` and then search for TODO(my_execution_environment) in the generated file to do the modifications.

Execution environment
------------------------
Each execution environment is defined as a subclass of :class:`gymnos.execution_environments.execution_environment.ExecutionEnvironment` implementing any methods needed to train a Gymnos experiment and overriding ``Config`` class variable to define the configuration variables for this execution environment.

my_execution_environment.py
----------------------------

``my_execution_environment.py`` first look like this:

.. code-block:: python

    #
    #
    #   MyExecutionEnvironment
    #
    #

    from .. import config
    from .execution_environment import ExecutionEnvironment

    class MyExecutionEnvironment(ExecutionEnvironment):
        """
        TODO(my_execution_environment): Description of my execution environment
        """

        class Config(config.Config):
            """
            OPTIONAL(my_execution_environment)
            """

        def __init__(self, trainer, config_files=None):
            super().__init__(trainer, config_files=config_files)

        def train(self):
            """
            TODO(my_execution_environment): Train experiment
            """

        def monitor(self, **train_kwargs):
            """
            OPTIONAL(my_execution_environment): Monitor training status
            """


Training experiment
===========================

Override ``train()`` method to execute experiment in execution environment.
It returns a dictionnary with training outputs, for example, the execution id.

.. code-block:: python

    def train(self):
        experiment_json = self.trainer.to_dict()

        execution_id = run_experiment_in_environment(experiment_json)

        return {"execution_id": execution_id}

Monitoring training
============================

If you want to monitor the training status, for example, printing logs, override ``monitor`` method. The arguments for this method will be the ``kwargs`` from your ``train`` output.

.. code-block:: python

    def monitor(self, execution_id):
        while not completed:
            completed, logs = fetch_logs_for(execution_id)

            print(logs)

            time.sleep(60)

Specifying ``Config``
=========================

Use the ``Config`` class to define your required and optional variables. The user will specify these values using environment variables or using a configuration located at ~/.gymnos/gymnos.json

.. code-block:: python

    class MyExecutionEnvironment(ExecutionEnvironment):

        class Config(config.Config):

            MY_EXECUTION_ENVIRONMENT_USERNAME = config.Value(required=True, help="Username for execution environment")


Then, you can use the values for this variables in your methods using ``self.config``. If the user has not provided all required variables, an exception is thrown:

.. code-block:: python

    def train(self):
        login_to_environment(
            user=self.config.MY_EXECUTION_ENVIRONMENT_USERNAME,
            password=self.config.MY_EXECUTION_ENVIRONMENT_PASSWORD
        )

Summary
============

1. Create ``MyExecutionEnvironment`` in ``gymnos/execution_environments/my_execution_environment.py`` inheriting from :class:`gymnos.execution_environments.execution_environment.ExecutionEnvironment`
2. Define configuration variables overriding class variable ``Config``
3. Implement methods to train and monitor experiments

Adding the execution environment to ``Telefonica/gymnos``
============================================================

If you'd like to share your work with the community, you can check in your execution environment implementation to Telefonica/gymnos. Thanks for thinking of contributing!

Before you send your pull request, follow these last few steps (check :ref:`contributing` to see more details):

1. Test execution environment with any Gymnos experiment
-----------------------------------------------------------
Check that your execution environment is working with a Gymnos experiment.

2. Add documentation
----------------------
Add execution environment documentation.

3. Check your code style
--------------------------
Follow the `PEP8 Python style guide <https://www.python.org/dev/peps/pep-0008/>`_, except Gymnos uses 120 characters as maximum line length.

You can lint files running ``flake8`` command:

.. code-block:: console

    $ flake8

Adding the execution environment from other repository
================================================================

You can also add an execution environment from other repository in a very simple way by converting your repository into a Python library.

Once you have defined your ``setup.py``, create and register your Gymnos execution environments in the same way we have shown.

Here is a minimal example. Say we have our library named ``gymnos_my_execution_environments`` and we want to add the execution environment ``my_execution_environment``. You have to:

1. Create ``MyExecutionEnvironment`` in ``gymnos_my_execution_environments/my_execution_environment.py`` inheriting from :class:`gymnos.execution_environments.execution_environment.ExecutionEnvironment` and implementing the abstract methods.
2. Register your execution environment in your module ``__init__.py`` referencing the type and the path:

.. code-block:: python
    :caption: gymnos_my_execution_environments/__init__.py

    import gymnos

    gymnos.execution_environments.register(
        type="my_execution_environment",
        entry_point="gymnos_my_execution_environments.my_execution_environment.MyExecutionEnvironment"
    )

That's it, when someone wants to run ``my_execution_environment`` from ``gymnos_my_execution_environments``, simply ``pip install`` the package and reference the package when you are loading the execution environment with the following format: ``<module_name>:<execution_environment>``.

For example:

.. code-block:: python

    gymnos.execution_environments.load("gymnos_my_execution_environments:my_execution_environment")
