New dataset
==============================

In this tutorial we will create a new dataset.

The name of the dataset will be ``my_dataset``.

First of all, we will run the command :ref:`gymnos-create`:

.. prompt:: bash

    gymnos-create dataset my_dataset

This will create the Python module ``gymnos/datasets/my_dataset`` with the following files:

    - ``__init__.py``: entrypoint for the dataset. It contains the docstring for the module.
    - ``__dataset__.py``: gymnos configuration for dataset
    - ``dataset.py``: dataset class. Here we will define the dataset
    - ``hydra_conf.py``: Hydra configuration definition. Here we will define the parameters for the dataset.

Adding the docstring
-----------------------

First of all, we will modify the docstring for the module:

.. code-block:: python
    :caption: __init__.py

    """
    Small description about the dataset
    """

Defining the parameters
------------------------

Now we will define the dataset parameters using a `dataclass <https://docs.python.org/3/library/dataclasses.html>`_.

We will add two parameters as example:

    - A required string parameter named ``param_1``
    - An optional int parameter named ``param_2`` with default value: ``2``

.. code-block:: python
    :caption: hydra_conf.py

    from dataclasses import dataclass, field


    @dataclass
    class MyDatasetHydraConf:

        param_1: str
        param_2: int = 2


        _target_: str = field(init=False, default="gymnos.datasets.my_dataset.dataset.MyDataset")

The ``_target_`` parameter is mandatory and must default to the path of the dataset. This will be automatically defined by ``gymnos-create``. It is used by `Hydra <https://hydra.cc/docs/next/advanced/instantiate_objects/overview/>`_.

Defining the dataset
---------------------

First of all, we will write a class docstring explaining about the data structure and class parameters.

.. code-block:: python
    :caption: dataset.py

    @dataclass
    class MyDataset(MyDatasetHydraConf, BaseDataset):
        """
        Data has the following structure:

        .. code-block::

            index.csv  # CSV file with 3 columns (row_1, row_2, row_3)
            data/
                class1.json  # data for class1 with the following keys -> age, gender
                class2.json  # data for class2 with the following keys -> age, gender

        Parameters
        -------------
        param_1:
            Description about param_1
        param_2:
            Description about param_2
        """

Then we will implement the ``__call__(root)`` method where root is the directory where dataset files are stored.

.. code-block:: python
    :caption: dataset.py

    from ...utils import extract_archive


    class MyDataset(MyDatasetHydraConf, BaseDataset):

        def __call__(root):
            download_dir = SOFIA.download_dataset("johndoe/datasets/my-dataset")  # download all files

            extract_archive(os.path.join(download_dir, "data.zip"), os.path.join(root, "data"))  # we will extract data.zip to root/data directory

            if not os.path.isfile(os.path.join(root, "index.csv")):
                os.symlink(os.path.join(download_dir, "index.csv"), os.path.join(root, "index.csv"))  # instead of copying the file, we will symlink the file to optimize storage

Running the dataset
--------------------

Once finished, you can run your dataset with Hydra using the command ``gymnos-train``:

.. prompt:: bash

    gymnos-train dataset=my_dataset dataset.param_1="example string" dataset.param_2=5

.. tip::
    You can use ``trainer=dummy`` to check that your dataset is working properly, e.g:

    .. prompt:: bash

        gymnos-train dataset=my_dataset trainer=dummy



Documentation
---------------

Remember to check the :ref:`documentation` for your new dataset
