########################
Execution Environments
########################

Usage
***********

.. code-block:: python

    fourth_platform = gymnos.execution_environments.load("fourth_platform", trainer=trainer)

    info = fourth_platform.train()  # execute training in 4th platform

    fourth_platform.monitor(**info)  # monitor training from 4th platform


All Execution Environments
****************************
.. contents::
    :local:

fourth_platform
========================
.. autoclass:: gymnos.execution_environments.fourth_platform.FourthPlatform
    :noindex:
    :members:
    :exclude-members: Config

    .. autoclass:: gymnos.execution_environments.fourth_platform.FourthPlatform.Config
        :noindex:
        :members:

wondervision
========================
.. autoclass:: gymnos.execution_environments.wondervision.WonderVision
    :noindex:
    :members:
    :exclude-members: Config

