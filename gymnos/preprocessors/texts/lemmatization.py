#
#
#   Lemmatization Preprocessor
#
#


from ..preprocessor import Preprocessor
from ...utils.iterator_utils import apply
from ..utils.spacy import get_spacy_nlp


class Lemmatization(Preprocessor):
    """
    Lemmatize words.

    Parameters
    ----------
    language: str, optional
        Text language. The current available languages are the following:

            - ``"english"``
            - ``"spanish"```
    """

    def __init__(self, language="english"):
        if language == "english":
            self.nlp = get_spacy_nlp("en")
        elif language == "spanish":
            self.nlp = get_spacy_nlp("es")
        else:
            raise ValueError("Language not supported")

    def fit(self, X, y=None):
        return self

    def fit_generator(self, generator):
        return self

    def _transform_sample(self, x):
        doc = self.nlp(str(x))
        return " ".join([token.lemma_ for token in doc])

    def transform(self, X):
        return apply(X, self._transform_sample)

    def save(self, save_dir):
        pass

    def restore(self, save_dir):
        pass
