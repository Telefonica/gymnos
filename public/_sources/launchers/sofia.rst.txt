.. _sofia_launcher:

SOFIA
==============================

.. code-block:: console

    $ gymnos-train -m hydra/launcher=sofia hydra.launcher.<ANY_SOFIA_HYDRA_LAUNCHER_PARAM>

.. list-table::
   :header-rows: 1

   * - Parameter
     - Required?
     - Default
     - Description
   * - project_name
     - .. raw:: html

        <p style='text-align: center'>&#11093;</p>
     -
     - SOFIA project name to associate training
   * - ref
     -
     - ``<HEAD>``
     - Gymnos release, branch or commit. It will be the environment where training will be executed. If command is executed on gymnos git directory, it will try to infer the current commit.
   * - device
     -
     - ``"CPU"``
     - Device to execute training. One of the following: ``"CPU"``, ``"GPU"``
