#
#
#   Preprocessor
#
#

import dill

from tqdm import tqdm
from abc import ABCMeta, abstractmethod


class Preprocessor(metaclass=ABCMeta):
    """
    Base class for all Gymnos preprocessors.

    You need to implement the following methods: ``fit`` and optionally ``transform``.
    """

    @abstractmethod
    def fit(self, X, y=None):
        """
        Fit preprocessor to training data.

        Parameters
        ----------
        X: array_like
            Features
        y: array_like, optional
            Labels

        Returns
        --------
        self: Preprocessor
            Own instance for chain purposes.
        """

    def fit_generator(self, generator):
        """
        Fit preprocessors to generator.

        Parameters
        ----------
        generator: generator
            Generator iterating X and y

        Returns
        -------
        self: Preprocessor
            Own instance for chain purposes.
        """
        raise NotImplementedError(("Preprocessor {} don't implement fit_generator " +
                                   "method").format(self.__class__.__name__))

    @abstractmethod
    def transform(self, X):
        """
        Transform data.

        Parameters
        ----------
        X: array_like
            Features

        Returns
        -------
        X_t: array_like
            Transformed features
        """


class Pipeline:

    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors or []

    def reset(self):
        self.preprocessors = []

    def add(self, preprocessor):
        self.preprocessors.append(preprocessor)

    def fit(self, X, y=None):
        if not self.preprocessors:
            return self

        pbar = tqdm(self.preprocessors)
        for preprocessor in pbar:
            pbar.set_description("Fitting with {}".format(preprocessor.__class__.__name__))
            preprocessor.fit(X, y)
            X = preprocessor.transform(X)

        return self

    def fit_generator(self, generator):
        if not self.preprocessors:
            return self

        fitted_preprocessors = []
        pbar = tqdm(self.preprocessors)
        for preprocessor in pbar:
            def generator_with_transform():
                for X, y in generator:
                    for fitted_preprocessor in fitted_preprocessors:
                        X = fitted_preprocessor.transform(X)
                    yield X, y

            pbar.set_description("Fitting generator with {}".format(preprocessor.__class__.__name__))
            preprocessor.fit_generator(generator_with_transform())
            fitted_preprocessors.append(preprocessor)

        return self

    def transform(self, X):
        if not self.preprocessors:
            return X

        for preprocessor in self.preprocessors:
            X = preprocessor.transform(X)

        return X

    def save(self, save_path):
        with open(save_path, "wb") as pickle_file:
            dill.dump(self.preprocessors, pickle_file)

    def restore(self, save_path):
        with open(save_path, "rb") as pickle_file:
            self.preprocessors = dill.load(pickle_file)

    def __str__(self):
        preprocessors_names = [prep.__class__.__name__ for prep in self.preprocessors]
        return " | ".join(preprocessors_names)

    def __len__(self):
        return len(self.preprocessors)
