.. _sofia_launcher:

SOFIA
==============================

.. prompt:: bash

    gymnos-train -m hydra/launcher=sofia hydra.launcher.<ANY_SOFIA_HYDRA_LAUNCHER_PARAM>

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
   * - device
     - .. raw:: html

        <p style='text-align: center'>&#11093;</p>
     -
     - Device to execute training. One of the following: ``"CPU"``, ``"GPU"``
   * - ref
     -
     - ``<HEAD>``
     - Gymnos release, branch or commit. It will be the environment where training will be executed. If command is executed on gymnos git directory, it will try to infer the current commit.
   * - notify_on_completion
     -
     - false
     - Whether or not send email when job has completed
