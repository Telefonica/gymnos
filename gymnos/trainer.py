#
#
#   Base class for trainer
#
#

from abc import ABCMeta, abstractmethod


class Trainer(metaclass=ABCMeta):

    def setup(self, data_dir):
        pass

    @abstractmethod
    def train(self):
        ...

    def test(self):
        raise NotImplementedError(f"Trainer {self.__class__.__name__} does not support test")
