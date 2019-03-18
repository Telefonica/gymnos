#
#
#   Preprocessor
#
#


class Preprocessor:

    def transform(self, X):
        raise NotImplementedError()


class PreprocessorStack:

    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors or []


    def reset(self):
        self.preprocessors = []


    def add(self, preprocessor):
        self.preprocessors.append(preprocessor)

    def transform(self, X):
        for preprocessor in self.preprocessors:
            X = preprocessor.transform(X)

        return X

    def __len__(self):
        return len(self.preprocessors)
