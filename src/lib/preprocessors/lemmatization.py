#
#
#   Lemmatization Preprocessor
#
#


from .preprocessor import Preprocessor
from ..utils.iterator_utils import apply
from ..utils.spacy_utils import get_spacy_nlp


class Lemmatization(Preprocessor):

    def __init__(self, language="en"):
        if language == "english":
            self.nlp = get_spacy_nlp("en")
        elif language == "spanish":
            self.nlp = get_spacy_nlp("es")
        else:
            raise ValueError("Language not supported")


    def __transform_sample(self, x):
        doc = self.nlp(x)
        return " ".join([token.lemma_ for token in doc])


    def transform(self, X):
        return apply(X, self.__transform_sample)
