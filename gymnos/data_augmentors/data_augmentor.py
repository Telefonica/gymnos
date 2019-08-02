# data_augmentor.py
# Author: Marcus D. Bloice <https://github.com/mdbloice> and contributors
# Licensed under the terms of the MIT Licence.
"""
The DataAugmentors module contains classes for all operations used by Augmentor.

The classes contained in this module are not called or instantiated directly
by the user, instead the user interacts with the
:class:`~Augmentor.Pipeline.Pipeline` class and uses the utility functions contained
there.

In this module, each operation is a subclass of type :class:`DataAugmentor`.
The :class:`~Augmentor.Pipeline.Pipeline` objects expect :class:`DataAugmentor`
types, and therefore all operations are of type :class:`DataAugmentor`, and
provide their own implementation of the :func:`~DataAugmentor.transform`
function.

Hence, the documentation for this module is intended for developers who
wish to extend Augmentor or wish to see how operations function internally.

For detailed information on extending Augmentor, see :ref:`extendingaugmentor`.
"""
import random

from abc import ABCMeta, abstractmethod


class DataAugmentor(metaclass=ABCMeta):
    """
    The class :class:`DataAugmentor` represents the base class for all operations
    that can be performed. Inherit from :class:`DataAugmentor`, overload
    its methods, and instantiate super to create a new operation. See
    the section on extending Augmentor with custom operations at
    :ref:`extendingaugmentor`.
    """

    def __init__(self, probability):
        """
        All operations must at least have a :attr:`probability` which is
        initialised when creating the operation's object.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :type probability: Float
        """
        self.probability = probability

    def __str__(self):
        """
        Used to display a string representation of the operation, which is
        used by the :func:`Pipeline.status` to display the current pipeline's
        operations in a human readable way.

        :return: A string representation of the operation. Can be overridden
         if required, for example as is done in the :class:`Rotate` class.
        """
        return self.__class__.__name__

    @abstractmethod
    def transform(self, item):
        """
        Perform the operation on the passed images. Each operation must at least
        have this function, which accepts an image as a numPy array.

        :param image: The image(s) to transform.
        :type image: np.array
        :return: The transformed image
        """
        raise RuntimeError("Illegal call to base class.")


class Pipeline:

    def __init__(self, data_augmentors=None):
        self.data_augmentors = data_augmentors or []

    def reset(self):
        self.data_augmentors = []

    def add(self, data_augmentor):
        self.data_augmentors.append(data_augmentor)

    def transform(self, item):
        for data_augmentor in self.data_augmentors:
            r = round(random.uniform(0, 1), 1)
            if r <= data_augmentor.probability:
                item = data_augmentor.transform(item)
        return item

    def __str__(self):
        data_augmentors_names = [str(da) for da in self.data_augmentors]
        return " | ".join(data_augmentors_names)

    def __len__(self):
        return len(self.data_augmentors)
