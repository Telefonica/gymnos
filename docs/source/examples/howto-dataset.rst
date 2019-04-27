#####################################
How to create a Dataset
#####################################

Implementing a dataset in Gymnos is really simple, just inherit from ``ClassificationDataset`` or ``RegressionDataset`` and overwrite some methods.

.. note::
    The training configuration (:class:`lib.core.dataset.Dataset`) will read ``lib.var.datasets.json`` to find the dataset given the dataset name. If you want to add a dataset, give it a name and add the location of the dataset.


Dataset
------------------
.. autoclass:: lib.datasets.dataset.Dataset
    :members:
    :inherited-members:

Regression dataset
==================
.. autoclass:: lib.datasets.dataset.RegressionDataset
    :members:
    :inherited-members:

Classification dataset
======================
.. autoclass:: lib.datasets.dataset.ClassificationDataset
    :members:
    :inherited-members:


Mixins
------

We provide `mixins <https://www.ianlewis.org/en/mixins-and-python>`_ with default functionality for dataset methods.

Kaggle
======

.. autoclass:: lib.datasets.mixins.kaggle.KaggleMixin
    :members: download

Public URL
==========


.. autoclass:: lib.datasets.mixins.public_url.PublicURLMixin
    :members: download
