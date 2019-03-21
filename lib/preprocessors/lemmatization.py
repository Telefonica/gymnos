#
#
#   Lemmatization Preprocessor
#
#

import spacy

from .preprocessor import Preprocessor
from ..utils.iterator_utils import apply


class Lemmatization(Preprocessor):

    def __init__(self, language="english"):
        if language == "spanish":
            self.nlp = spacy.load("es")
        else:
            self.nlp = spacy.load("en")


    def __transform_sample(self, x):
        doc = self.nlp(x)
        return " ".join([token.lemma_ for token in doc])


    def transform(self, X):
        return apply(X, self.__transform_sample)
