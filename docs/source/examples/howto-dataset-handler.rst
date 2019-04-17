#####################################
How to create a Dataset Handler
#####################################

To create a dataset you need to inherit from ``Dataset`` or its subclasses and implement some methods.
If you want to use the dataset in an experiment you must add the dataset location with and id in ``lib.var.datasets.json``, e.g ``mydataset: lib.datasets.mydataset.MyDataset``.

Base dataset
===============

This is the parent dataset that all classes must inherit.
The ``Dataset`` class is defined in ``lib.datasets.dataset`` and you must implement the following methods:

.. code-block:: python

   def download(self, download_dir):
      # download files needed to read dataset to download_dir
      ...

   def read(self, download_dir):
      # read files downloaded and return features and labels (must be one-hot encoded)
      ...
      return X, y


An example of dataset is the following:

.. code-block:: python

    import os
    import requests
    import pandas as pd

    from .dataset import Dataset

    from keras.utils import to_categorical


    class MyDataset(Dataset):

        def download(self, download_dir):
            res = requests.get("http://example-dataset.com")
            with open("dataset.csv", "wb") as outfile:
                outfile.write(res.content)

        def read(self, download_dir):
            df = pd.read_csv(os.path.join(download_dir, "example.csv"))
            X = df.drop(columns="label")
            y = df.label

            return X, to_categorical(y)


Kaggle dataset
===============

If you want to implement a Kaggle dataset, you only need to inherit from ``lib.datasets.dataset.KaggleDataset``, fill the ``kaggle_dataset_name`` class variable with the id of the Kaggle dataset and implement the ``read`` method. Optionally, if you want to download only some files of the Kaggle dataset, fill the ``kaggle_dataset_files`` with the filenames you want to download.

.. code-block:: python

    from .dataset import KaggleDataset

    class MyKaggleDataset(KaggleDataset):

        kaggle_dataset_name = "lava18/google-play-store-apps"
        kaggle_dataset_files = ["googleplaystore.csv"]

        def read(self, download_dir):
            ...
            return X, y

Public dataset
===============

If you want to implement a public dataset downloadable through URL, you only need to inherit from ``lib.datasets.dataset.PublicDataset``, fill the ``public_dataset_files`` class variable with the URLs you want to download and implement the ``read`` method.

.. code-block:: python

    from .dataset import PublicDataset

    class MyPublicDataset(PublicDataset):

        public_dataset_files = "https://fred.stlouisfed.org/graph/fredgraph.csv"

        def read(self, download_dir):
            ...
            return X, y


Library dataset
===============

If you want to implement a dataset that belongs to a library like ``Keras`` or ``Scikit-Learn``, you only need to inherit from ``lib.datasets.dataset.LibraryDataset`` and implement the ``read`` method.

.. code-block:: python

    from .dataset import LibraryDataset

    from keras.utils import to_categorical
    from sklearn.datasets import load_iris


    class MyLibraryDataset(LibraryDataset):

        def read(self, download_dir):
            X, y = load_iris(return_X_y=True)
            return X, to_categorical(y)
