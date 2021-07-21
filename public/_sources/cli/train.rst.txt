.. _gymnos-train:

gymnos-train
==============================

This is the `Hydra <https://hydra.cc>`_ entrypoint for trainings.

.. code-block:: console

    $ gymnos-train trainer=<TRAINER> trainer.<PARAM>=<VALUE> dataset=<DATASET> dataset.<PARAM>=<VALUE> test=false verbose=true dependencies.install=false mlflow.run_name=null mlflow.experiment_name=Default mlflow.log_trainer_params=true

.. list-table::
   :header-rows: 1

   * - Parameter
     - Required?
     - Default
     - Description
   * - trainer
     - .. raw:: html

        <p style='text-align: center'>&#11093;</p>
     -
     - Trainer to use, e.g ``vision.image_classification.transfer_efficientnet``
   * - dataset
     - .. raw:: html

        <p style='text-align: center'>&#11093;</p>
     -
     - Dataset to use, e.g ``dogs_vs_cats``
   * - trainer.<PARAM>
     -
     -
     - Override trainer parameter <PARAM>
   * - dataset.<PARAM>
     -
     -
     - Override dataset parameter <PARAM>
   * - test
     -
     - ``false``
     - Whether or not test model at the end of the training
   * - verbose
     -
     - ``true``
     - Verbosity.
   * - dependencies.install
     -
     - ``false``
     - Whether or not automatically install model dependencies
   * - mlflow.run_name
     -
     - ``null``
     - MLFlow run name
   * - mlflow.experiment_name
     -
     - ``"Default"``
     - MLFlow experiment name
   * - mlflow.log_trainer_params
     -
     - ``true``
     - Whether or not log trainer parameters to MLFlow


.. note::

    To have tab completion, install the following plugin. More information at `Hydra tab completion <https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion/>`_

    .. asciinema:: ../_static/asciinema/tab-completion.cast

    .. tabs::

        .. tab:: bash

            .. code-block:: console

                $ eval "$(gymnos-train -sc install=bash)"

        .. tab:: zsh

            .. code-block:: console

                $ eval "$(gymnos-train -sc install=bash)"

        .. tab:: fish

            .. code-block:: console

                $ gymnos-train -sc install=fish | source
