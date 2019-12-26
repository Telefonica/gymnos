####################
Add a dataset
####################

Overview
==========
Datasets are distributed in all kinds of formats and in all kinds of places, and they're not 
always stored in a format that's ready to feed into a machine learning pipeline. Enter Gymnos datasets.

Gymnos datasets provides a way to transform all those datasets into a standard format and do the preprocessing necessary to make them ready for a machine learning pipeline.

To enable this, each dataset implements a subclass of :class:`gymnos.datasets.Dataset`, which specifies:

* Where the data is coming from (i.e its URL) 
* What the dataset looks like (i.e it features)
* And the individual records in the dataset

The first time a dataset is used, the dataset is downloaded, prepared and written to disk. Subsequent access will read from those preprocessed files directly.

Writing ``my_dataset.py``
==========================

Use the default template
-------------------------
If you want to :ref:`cofntribute to our repo <contributing>` and add a new dataset, the following script will help you get started generating the required python files. To use it, clone the `Gymnos <https://github.com/Telefonica/gymnos>`_ repository and run the following command:

.. code-block:: console

  $ python3 -m scripts.create_new dataset --name my_dataset

This command will create ``gymnos/datasets/my_dataset.py``, and modify ``gymnos/__init__.py`` to register dataset so we can load it using ``gymnos.load``.

The dataset registration process is done by associating the dataset name with their path:

.. code-block:: python
    :caption: gymnos/__init__.py

    datasets.register(
        name="my_dataset",
        entry_point="gymnos.datasets.my_dataset.MyDataset"
    )

Go to ``gymnos/datasets/my_dataset.py`` and then search for TODO(my_dataset) in the generated file to do the modifications.

Dataset
--------
Each dataset is defined as a subclass of :class:`gymnos.datasets.Dataset` implementing the following methods / properties:

* ``features_info`` and ``labels_info``: describes the features and labels of your dataset
* ``download_and_prepare``: downloads the source data.
* ``__getitem__`` or ``__iter__``: returns a single row of data
* ``__len__``: returns the dataset length

my_dataset.py
---------------

``my_dataset.py`` first look like this:

.. code-block:: python

    #
    #
    #   MyDataset
    #
    #

    from .dataset import Dataset, DatasetInfo, Array, ClassLabel

    class MyDataset(Dataset):
        """
        TODO(my_dataset): Description of my dataset.
        """
        @property
        def features_info(self):
            # {TODO}: Specifies the information about the features (shape, dtype, etc...)

        @property
        def labels_info(self):
            # {TODO}: Specifies the information about the labels (shape, dtype, etc ...)

        def download_and_prepare(self, dl_manager):
            pass # TODO(my_dataset): download any file you will need later in the __getitem__ and __len__ function

        def __getitem__(self, given):
            pass # TODO(my_dataset): Get dataset item/s. Given can be a slice object or an int. Called after download_and_prepare.

        def __len__(self):
            pass # TODO(my_dataset): Dataset length. Called after download_and_prepare


Downloading and extracting source data
=======================================

Most datasets need to download data from the web. All downloads and extractions must go through the :class:`~gymnos.services.download_manager.DownloadManager`. 

For example, one can download URLs with ``http`` service using their :class:`~gymnos.services.download_manager.DownloadManager.download` method and extract files with :class:`~gymnos.services.download_manager.DownloadManager.extract` method:

.. code-block:: python

    def download_and_prepare(self, dl_manager):
        dl_paths = dl_manager["http"].download({
            "foo": "https://example.com/foo.zip",
            "bar": "https://example.com/bar.zip",
        })

        self.edl_paths = dl_manager.extract(dl_paths)

        self.edl_paths["foo"], self.edl_paths["bar"]

Specifying ``features_info`` and ``labels_info``
====================================================

:class:`gymnos.datasets.DatasetInfo` describes the dataset.

You need to specify the shape and dtype for your features and labels using the ``Array`` class.
If you have class labels, specify them using ``ClassLabel`` type.

.. code-block:: python

    from .dataset import Dataset, Array, ClassLabel

    class MyDataset(Dataset):

        @property
        def features_info(self):
            return Array(shape=[80, 80], dtype=np.uint8)

        @property
        def labels_info(self):
            return ClassLabel(names=["dog", "cat"])


Specifying length of your dataset
===================================

To specify the number of samples of your dataset. Implement the ``__len__`` method. This method will always be called after ``download_and_prepare``.

.. code-block:: python

    def __len__(self):
        return len(self.edl_paths["foo"])

Returning rows of data
============================

You can return rows of data in two different ways:

- Mapping indices to rows
- Iterating over rows
- Loading Spark DataFrame

.. note::

    **Which one should I use?**
    Training datasets that maps indices to rows will always be more performant due to the possibility of multiprocessing and how the splitting works. If your dataset allows it, map indices to rows.

The dataset must return two values: the features or ``X`` and the labels or ``y`` for each row of data.
The allowed data types for your features are the following:

- A string or a number
- An array-like e.g a list, a tuple or a set.
- A NumPy array
- A Pandas Series

The allowed data types for your labels depends on the problem you're trying to solve. For classification tasks, you must return the class indices, e.g for 2 classes return 0 or 1. For regression tasks, you can return a number or an array of numbers.

Mapping Dataset
------------------
This dataset maps indices to rows. Just implement the ``__getitem__`` returning the corresponding row to the given index.

.. code-block:: python

    def __getitem__(self, index):
        image_path = self.image_path_[index]
        ...
        return img_arr, label


Iterating Dataset
-------------------
Some datasets does not allow to retrieve rows by index without fully loading dataset into memory. To solve this issue, you can iterate over rows of your dataset.
Instead of inheriting from ``Dataset``, you must inherit from ``IterableDataset`` class, and implement the ``__iter__`` yielding rows of data.

.. code-block:: python

    def __iter__(self):
        for row in iterate_data():
            yield row

Spark Dataset
-----------------
To create a distributed Spark dataset, instead of inheriting from ``Dataset``, you must inherit from ``SparkDataset`` class and implement the ``load`` returning the DataFrame.

The constructor for this dataset will be the column name for features and the column name for labels:

.. code-block:: python

    class MySparkDataset(SparkDataset):

        def __init__(self, features_col="features", labels_col="labels"):
            self.features = features_col
            self.labels_col = labels_col

To load the DataFrame, implement the ``load`` method:

.. code-block:: python

    def load(self):
        df = self.spark.read.csv("mydata.csv", header=True, inferSchema=True)  # you can access SparkSession using self.spark
        ...  # do any basic preprocessing to clean your data
        return df


Summary
=============
1. Create ``MyDataset`` in ``gymnos/dataset/my_dataset.py`` inheriting from :class:`gymnos.datasets.dataset.Dataset` if your dataset can map indices to rows or :class:`gymnos.datasets.dataset.IterableDataset` if your dataset iterates over rows of data and implement the following properties:

* ``features_info``
* ``labels_info``

With the following abstract methods:

* ``download_and_prepare(dl_manager)``
* ``__len__()``

If your dataset inherits from :class:`gymnos.datasets.dataset.Dataset`, write the following method:

* ``__getitem__(index)``

If your dataset inherits from :class:`gymnos.datasets.dataset.IterableDataset`, write the following method:

* ``__iter__()``

If your dataset inherits from :class:`gymnos.datasets.dataset.SparkDataset`, write the following method:

* ``load()``

2. Register the dataset in ``gymnos/__init__.py`` by adding:

.. code-block:: python

    datasets.register(
        name="my_dataset",
        entry_point="gymnos.datasets.my_dataset.MyDataset"
    )


Adding the dataset to ``Telefonica/gymnos``
===========================================

If you'd like to share your work with the community, you can check in your dataset implementation to Telefonica/gymnos. Thanks for thinking of contributing!

Before you send your pull request, follow these last few steps (check :ref:`contributing` to see more details):

1. Run ``download_and_prepare`` locally
----------------------------------------
Run ``download_and_prepare`` locally to ensure that data generation works.

2. Add documentation
----------------------
Add dataset documentation.

3. Run tests
-------------
Execute the following command to run automated tests:

.. code-block:: console

    $ pytest

4. Check your code style
--------------------------
Follow the `PEP8 Python style guide <https://www.python.org/dev/peps/pep-0008/>`_, except Gymnos uses 120 characters as maximum line length.

You can lint files running ``flake8`` command:

.. code-block:: console

    $ flake8


Adding the dataset from other repository
=================================================

You can also add a dataset from other repository in a very simple way by converting your repository into a Python library.

Once you have defined your ``setup.py``, create and register your Gymnos datasets in the same way we have shown.

Here is a minimal example. Say we have our library named ``gymnos_my_datasets`` and we want to add the dataset ``my_dataset``. You have to:

1. Create ``MyDataset`` in ``gymnos_my_datasets/my_dataset.py`` inheriting from :class:`gymnos.datasets.dataset.Dataset` and implementing the abstract methods
2. Register dataset in your module ``__init__.py`` referencing the name and the path:

.. code-block:: python
    :caption: gymnos_my_datasets/__init__.py

    import gymnos

    gymnos.datasets.register(
        name="my_dataset",
        entry_point="gymnos_my_datasets.my_dataset.MyDataset"
    )


That's it, when someone wants to run ``my_dataset`` from ``gymnos_my_datasets``, simply ``pip install`` the package and reference the package when you are loading the dataset with the following format: ``<module_name>:<dataset_name>``.

For example:

.. code-block:: python

    gymnos.datasets.load("gymnos_my_datasets:my_dataset")
