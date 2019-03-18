#
#
# Transformer
#
#


class TransformerStack:

    def __init__(self, transformers=None):
        self.transformers = transformers or []


    def reset(self):
        self.transformers = []

    def add(self, transformer):
        self.transformers.append(transformer)

    def fit(self, X, y=None):
        X_fit = X

        for transformer in self.transformers:
            X_fit = transformer.fit_transform(X_fit)

        return self

    def transform(self, X, y=None):
        X_fit = X
        for transformer in self.transformers:
            X_fit = transformer.transform(X_fit)

        return X_fit

    def __len__(self):
        return len(self.transformers)
