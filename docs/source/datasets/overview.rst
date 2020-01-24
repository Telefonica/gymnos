Datasets
#########

Gymnos datasets are a collection of datasets with a common API allowing their use in a pipeline of a supervised learning system. All datasets inherit from :class:`gymnos.datasets.dataset.Dataset`.

Usage
*******
.. code-block:: python

    dataset = gymnos.datasets.load("dogs_vs_cats")

    dl_manager = gymnos.services.DownloadManager()
    dataset.download_and_prepare(dl_manager)   # download data and prepare dataset

    print(dataset.features_info)  # info about features

    print(dataset.labels_info)  # info about labels

    print(len(dataset))  # get number samples

    for x, y in dataset:  # iterate over dataset
      print(x, y)

    features, labels = dataset.load()  # load full dataset into memory

    dataset.to_hdf5("dogs_vs_cats.h5")  # guardar el dataset en un formato numericamente muy eficiente

    del dataset

    dataset = gymnos.datasets.load("hdf5", file_path="dogs_vs_cats.h5")  # restaurar dataset desde un archivo HDF5


All Datasets
*************

.. toctree::
   :maxdepth: 2
    
   ./audio.rst
   ./image.rst
   ./structured.rst
   ./text.rst
