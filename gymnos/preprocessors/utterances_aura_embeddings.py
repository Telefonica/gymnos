#
#
#   Utterances Aura Embeddings
#
#

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
        self.embeddings = lazy_imports.auracog_embeddings_embeddings.Embeddings(model_path=model_path, padding=False)
        self.normalizer = lazy_imports.auracog_utils_text.TextNormalizer('es_ES')

    def fit(self, X, y=None):
        return self

    def fit_generator(self, generator):
        return self

    def transform(self, X):
        result = []
        for sequence in X:
            sequence = eval(sequence.replace(' ', ','))
            normalized_sequence = self.__sequence_normalizer(sequence)
            list_embeddings = self.embeddings.transform(normalized_sequence)
            result.append(list_embeddings)
        return result

    def __sequence_normalizer(self, sequence):
        """ Simple pre-processing: lowercasing, remove punctuation and special characters and acents """
        total = []
        for phrase in sequence:
            total.append([val for val in self.normalizer.to_tkn(phrase).norm])
        return total

    def save(self, save_dir):
        pass

    def restore(self, save_dir):
        pass
