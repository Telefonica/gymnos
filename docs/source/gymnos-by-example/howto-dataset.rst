#####################################
How to create a Dataset
#####################################

Implementing a dataset in Gymnos is really simple, just inherit from :class:`gymnos.datasets.dataset.Dataset` and overwrite some methods.

.. note::
    The training configuration (:class:`gymnos.core.dataset.Dataset`) will read ``gymnos.var.datasets.json`` to find the dataset given the dataset's name. If you want to add a dataset, give it a name and add the dataset's location.


.. autoclass:: gymnos.datasets.dataset.Dataset
    :members:
    :special-members: __getitem__, __len__


Download Manager
------------------

.. autoclass:: gymnos.services.download_manager.DownloadManager
    :members:


Dataset Info
------------------

.. autoclass:: gymnos.datasets.dataset.DatasetInfo
    :members:

Array
^^^^^

.. autoclass:: gymnos.datasets.dataset.Array
    :members:


ClassLabel
"""""""""""

.. autoclass:: gymnos.datasets.dataset.ClassLabel
    :members:
    :show-inheritance:
