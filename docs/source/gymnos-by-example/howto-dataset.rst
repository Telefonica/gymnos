#####################################
How to create a Dataset
#####################################

Implementing a dataset in Gymnos is really simple, just inherit from :class:`lib.datasets.dataset.Dataset` and overwrite some methods.

.. note::
    The training configuration (:class:`lib.core.dataset.Dataset`) will read ``lib.var.datasets.json`` to find the dataset given the dataset's name. If you want to add a dataset, give it a name and add the dataset's location.


.. autoclass:: lib.datasets.dataset.Dataset
    :members:
    :special-members: __getitem__, __len__


Download Manager
------------------

.. autoclass:: lib.services.download_manager.DownloadManager
    :members:


Dataset Info
------------------

.. autoclass:: lib.datasets.dataset.DatasetInfo
    :members:

Array
^^^^^

.. autoclass:: lib.datasets.dataset.Array
    :members:


ClassLabel
"""""""""""

.. autoclass:: lib.datasets.dataset.ClassLabel
    :members:
    :show-inheritance:
