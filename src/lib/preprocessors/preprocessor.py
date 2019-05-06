#
#
#   Preprocessor
#
#

import dill

from tqdm import tqdm
from sklearn.base import TransformerMixin


class Preprocessor(TransformerMixin):
    """
    Base class for all Gymnos preprocessors.

    You need to implement the following methods: ``fit`` and optionally ``transform``.
    """

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
        return self

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
        raise NotImplementedError()


class Pipeline:

    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors or []

    def reset(self):
        self.preprocessors = []

    def add(self, preprocessor):
        self.preprocessors.append(preprocessor)

    def fit(self, X, y=None):
        if not self.preprocessors:
            return X

        pbar = tqdm(self.preprocessors)
        for preprocessor in pbar:
            pbar.set_description("Fitting with {}".format(preprocessor.__class__.__name__))
            preprocessor.fit(X, y)
            X = preprocessor.transform(X)

        return self

    def transform(self, X, data_desc=None):
        if not self.preprocessors:
            return X

        pbar = tqdm(self.preprocessors)
        for preprocessor in pbar:
            desc = "Transforming with {}".format(preprocessor.__class__.__name__)
            if data_desc is not None:
                desc += " ({})".format(data_desc)
            pbar.set_description(desc)

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
