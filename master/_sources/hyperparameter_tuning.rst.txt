Hyperparameter tuning
=======================

You can optimize hyperparameters easily with gymnos.

First of all, define the metric to optimize in your config file:

.. code-block:: yaml

    optimized_metric: val_loss

You can also define multiple metrics:

.. code-block:: yaml

    optimized_metric:
      - val_loss
      - val_acc

By default, the chosen metric will be the last one but you can also choose the minimum or the maximum:

.. code-block:: yaml

    optimized_metric:
      mode: max
      metric: val_acc

Now you will choose the algorithm to tune the hyperparameters:

    - `Optuna <https://hydra.cc/docs/plugins/optuna_sweeper/>`_
    - `Adaptive Experimentation Platform, aka Ax <https://hydra.cc/docs/plugins/ax_sweeper/>`_
    - `Nevergrad <https://hydra.cc/docs/plugins/nevergrad_sweeper/>`_

In this tutorial, we will use Optuna. For other libraries, check the documentation.

First of all, add the hydra sweeper in your config file:

.. code-block:: yaml

    defaults:
      - override /hydra/sweeper: optuna

Now we will customize the parameters for the algorithm:

.. code-block:: yaml

    hydra:
      sweeper:
        n_trials: 2
        direction: maximize
        search_space:
          trainer.num_train_timesteps:
            type: int
            low: 100_000
            high: 250_000
          trainer.use_rms_prop:
            type: categorical
            choices:
              - true
              - false

.. note::

    You can also tune hyperparameters with SOFIA but you will need to keep the terminal open.
    This is because although the trainings are run in SOFIA, the management of the hyperparameters is decided locally,
    i.e. your machine tells SOFIA which hyperparameters to test.
