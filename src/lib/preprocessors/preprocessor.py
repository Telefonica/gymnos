#
#
#   Preprocessor
#
#

from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin


class Preprocessor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
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

    def __len__(self):
        return len(self.preprocessors)
