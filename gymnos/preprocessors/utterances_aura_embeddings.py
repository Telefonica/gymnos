#
#
#   Utterances Aura Embeddings
#
#

import string

import unidecode

from .preprocessor import Preprocessor
from ..utils.lazy_imports import lazy_imports


class UtterancesAuraEmbeddings(Preprocessor):
    """
    For a list of sequence of sentences, firstly applies a preprocessing to every sequence
    (lowercasing, remove punctuation and special characters and acents). After that,
    transform the text in a embedding appling del model trained defined in model_path parameter.

    Parameters
    -----------
    model_path: str,
        path to lenguage model (.pkl) trained before.
    """

    def __init__(self, model_path):
        self.embeddings = lazy_imports.auracog_embeddings.Embeddings(model_path=model_path, padding=False)

    def fit(self, X, y=None):
        return self

    def fit_generator(self, generator):
        return self

    def transform(self, X):
        result = []
        for sequence in X:
            sequence = eval(sequence.replace(' ', ','))
            prep_sequence = UtterancesAuraEmbeddings._sequence_simple_preprocess(sequence)
            list_embeddings = self.embeddings.transform(prep_sequence)
            result.append(list_embeddings)
        return result

    @staticmethod
    def _sequence_simple_preprocess(sequence):
        """ Simple pre-processing: lowercasing, remove punctuation and special characters and acents """
        total = []
        string.punctuation += '“”«»¿¡‘’'
        table = str.maketrans({key: None for key in string.punctuation})
        for phrase in sequence:
            total.append([unidecode.unidecode(str(sent).lower().translate(table)) for sent in phrase.split(",")])
        return total

    def save(self, save_dir):
        pass

    def restore(self, save_dir):
        pass
