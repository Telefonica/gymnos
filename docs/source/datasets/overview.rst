Datasets
#########

Gymnos datasets are a collection of datasets with a common API allowing their use in a pipeline of a supervised learning system. All datasets inherit from :class:`gymnos.datasets.dataset.Dataset`.

Usage
*******
.. code-block:: python

    dataset = gymnos.datasets.load("dogs_vs_cats")

    print(dataset.info())  # info about features and labels

    dl_manager = gymnos.services.download_manager.DownloadManager()
    dataset.download_and_prepare(dl_manager)   # download data and prepare dataset

    print(len(dataset))  # get number samples

    print(dataset[0])  # get first sample

    data = dataset.as_numpy()   # load full dataset into memory

    dataset.to_hdf5("dogs_vs_cats.h5")  # guardar el dataset en un formato numericamente muy eficiente

    del dataset

    dataset = gymnos.datasets.dataset.HDF5Dataset("dogs_vs_cats.h5")   # restaurar dataset desde un archivo HDF5


All Datasets
*************

.. toctree::
   :maxdepth: 2

   ./image.rst
   ./structured.rst
   ./text.rst
