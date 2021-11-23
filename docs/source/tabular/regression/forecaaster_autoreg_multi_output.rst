.. _tabular.regression.forecaaster_autoreg_multi_output:

Forecaaster Autoreg Multi Output
================================

.. automodule:: gymnos.tabular.regression.forecaaster_autoreg_multi_output

.. prompt:: bash

    pip install gymnos[tabular.regression.forecaaster_autoreg_multi_output]

.. contents::
    :local:

.. _tabular.regression.forecaaster_autoreg_multi_output__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=tabular.regression.forecaaster_autoreg_multi_output

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.tabular.regression.forecaaster_autoreg_multi_output.trainer.ForecaasterAutoregMultiOutputTrainer
        :inherited-members:


.. _tabular.regression.forecaaster_autoreg_multi_output__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.tabular.regression.forecaaster_autoreg_multi_output import ForecaasterAutoregMultiOutputPredictor

    ForecaasterAutoregMultiOutputPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.tabular.regression.forecaaster_autoreg_multi_output.predictor.ForecaasterAutoregMultiOutputPredictor
   :members:
