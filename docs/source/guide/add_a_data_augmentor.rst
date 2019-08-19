######################
Add a data augmentor
######################

Overview
==============

Machine Learning data augmentors are distributed in all kinds of API specifications and in all kinds of places, and they're not always implemented in a format that's ready to feed into a machine learning pipeline. Enter Gymnos data augmentors.
Gymnos data augmentors provides a way to transform all those data augmentors into a standard format to make them ready for a machine learning pipeline.

To enable this, each data augmentor implements a subclass of :class:`gymnos.data_augmentors.DataAugmentor`, which specifies:

* How to transform images

Writing ``my_data_augmentor.py``
=================================

Use the default template
-------------------------
If you want to :ref:`contribute to our repo <contributing>` and add a new data_augmentor, the following script will help you get started generating the required python files. To use it, clone the `Gymnos <https://github.com/Telefonica/gymnos>`_ repository and run the following command:

.. code-block:: console

  $ python3 -m scripts.create_new data_augmentor --name my_data_augmentor

This command will create ``gymnos/data_augmentors/my_data_augmentor.py``,and modify ``gymnos/var/data_augmentors.json`` to reference data_augmentor name with their location so we can load it using ``gymnos.load``.

Go to ``gymnos/data_augmentors/my_data_augmentor.py`` and then search for TODO(my_data_augmentor) in the generated file to do the modifications.

DataAugmentor
---------------

Each data augmentor is defined as a subclass of :class:`gymnos.data_augmentors.DataAugmentor` implementing the following methods:

* ``transform``: perform the data augmentation operation on the input sample.

my_data_augmentor.py
---------------------

``my_data_augmentor.py`` first look like this:

.. code-block:: python

    #
    #
    #   MyDataAugmentor
    #
    #

    from .data_augmentor import DataAugmentor

    class MyDataAugmentor(DataAugmentor):
        """
        TODO(my_data_augmentor): Description of my data_augmentor.
        """

        def __init__(self, probability, **parameters):
            # TODO(my_data_augmentor): Define probability and initialize data augmentor parameters
            super().__init__(probability)

        def transform(self, image):
            # TODO(my_data_augmentor): Transform image

Specifying ``parameters``
==========================

Use the constructor to specify any parameters you need to build your data augmentor. These parameters may be required or optional although optional
parameters are preferable.

You must call ``super().__init__(probability)``.

.. code-block:: python

    class MyDataAugmentor(DataAugmentor):

        def __init__(self, probability, min_factor=0.1, max_factor=0.9):
            super().__init__(probability)

            self.min_factor = min_factor
            self.max_factor = max_factor


Transforming input samples
===========================

Implement this method to perform data augmentation on image.

.. code-block:: python

    def transform(self, image):
        ...
        return new_image

.. note::

    This method can't change the input shape.

Adding the data augmentor to ``Telefonica/gymnos``
===================================================

If you'd like to share your work with the community, you can check in your data augmentor implementation to Telefonica/gymnos. Thanks for thinking of contributing!

Before you send your pull request, follow these last few steps (check :ref:`contributing` to see more details):

1. Test data augmentor with any Gymnos image dataset
-----------------------------------------------------
Check that your data augmentor is working with a Gymnos image dataset.

2. Add documentation
----------------------
Add data augmentor documentation.

3. Check your code style
--------------------------
Follow the `PEP8 Python style guide <https://www.python.org/dev/peps/pep-0008/>`_, except Gymnos uses 120 characters as maximum line length.

You can lint files running ``flake8`` command:

.. code-block:: console

    $ flake8
