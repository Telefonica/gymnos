#
#
#   Sequences Aura Embeddings
#
#

from .preprocessor import Preprocessor
from ..utils.lazy_imports import lazy_imports


class SequencesAuraEmbeddings(Preprocessor):
    """
    For a list of sequence of sentences, firstly prepossesses to every sequence
    (lowercase, remove punctuation and special characters and accents). After that,
    transform the text in a embedding applying del model trained defined in model_path parameter.

    Parameters
    -----------
    model_path: str,
        path to language model (.pkl) trained before.
    """

    def __init__(self, model_path):
        self.embeddings = lazy_imports.auracog_embeddings_embeddings.Embeddings(model_path=model_path, padding=False)
        self.normalizer = lazy_imports.auracog_utils_text.TextNormalizer('es_ES')

    def fit(self, x, y=None):
        return self

    def fit_generator(self, generator):
        return self

    def transform(self, x):
        result = []
        for sequence in x:
            sequence = eval(sequence)
            normalized_sequence = self.__sequence_normalizer(sequence)
            list_embeddings = self.embeddings.transform(normalized_sequence)
            result.append(list_embeddings)
        return result

    def __sequence_normalizer(self, sequence):
        """ Simple pre-processing: lowercase, remove punctuation and special characters and accents """
        total = []
        for phrase in sequence:
            total.append([val for val in self.normalizer.to_tkn(phrase).norm])
        return total

    def save(self, save_dir):
        pass

    def restore(self, save_dir):
        pass
