#
#
# Transformer
#
#

from tqdm import tqdm

from ..utils.iterator_utils import count


class TransformerStack:

    def __init__(self, transformers=None):
        self.transformers = transformers or []


    def reset(self):
        self.transformers = []

    def add(self, transformer):
        self.transformers.append(transformer)

    def fit(self, X, y=None):
        for transformer in tqdm(self.transformers):
            transformer.fit(X, y)
            X = transformer.transform(X)

        return self

    def transform(self, X):
        if count(X) == 0:
            return X

        for transformer in self.transformers:
            X = transformer.transform(X)

        return X

    def __len__(self):
        return len(self.transformers)
