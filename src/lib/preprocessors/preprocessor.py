#
#
#   Preprocessor
#
#

from tqdm import tqdm
from sklearn.base import TransformerMixin


class Preprocessor(TransformerMixin):

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

    def __str__(self):
        preprocessors_names = [prep.__class__.__name__ for prep in self.preprocessors]
        return " | ".join(preprocessors_names)

    def __len__(self):
        return len(self.preprocessors)
