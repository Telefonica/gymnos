#
#
#   Sequences Aura Embeddings
#
#

import ast

import numpy as np

from .preprocessor import Preprocessor
from ..utils.lazy_imports import lazy_imports
auracog_embeddings = __import__(f"{lazy_imports.auracog_embeddings.__name__}.embeddings")
auracog_utils = __import__(f"{lazy_imports.auracog_utils.__name__}.text")


class SequencesAuraEmbeddings(Preprocessor):
    """
    For a list of sequence of sentences, firstly prepossesses to every sequence
    (lowercase, remove punctuation and special characters and accents). After that,
    transform the text in a embedding applying del model trained defined in model_path parameter.

    NOTE:
        - This preprocesor requires a module from a private PYPI server:

            auracog_utils==0.10.0 : https://github.com/Telefonica/aura-cognitive2-utils.git

    This preprocessor requires a module from a private PYPI server (artifactory/pypi-aura-repo).
     To add this module add the following line to ~/.pip/pip.conf:

    [global]
    extra-index-url = http://<username>:<password>@artifactory.hi.inet/artifactory/api/pypi/pypi-aura-cache/simple
    trusted-host = artifactory.hi.inet

    where <username> and <password> are your credentials from Telefonica.

    Installation: https://telefonicacorp.sharepoint.com/sites/AURA/Cognitive%20wiki/dev/utils2.aspx

    Parameters
    -----------
    model_path: str,
        path to language model (.pkl) trained before.
    """

    def __init__(self, model_path):
        self.embeddings = auracog_embeddings.embeddings.Embeddings(model_path=model_path, padding=False)
        self.normalizer = auracog_utils.text.TextNormalizer('es_ES')

    def fit(self, x, y=None):
        return self

    def fit_generator(self, generator):
        return self

    def transform(self, x):
        result = []
        for sequence in x:
            sequence = SequencesAuraEmbeddings.safe_eval(sequence)
            normalized_sequence = self.__sequence_normalizer(sequence)
            list_embeddings = self.embeddings.transform(normalized_sequence)
            result.append(list_embeddings)
        return np.array(result)

    def __sequence_normalizer(self, sequence):
        """ Simple pre-processing: lowercase, remove punctuation and special characters and accents """
        total = []
        for phrase in sequence:
            total.append([val for val in self.normalizer.to_tkn(phrase).norm])
        return total

    @staticmethod
    def safe_eval(s):
        try:
            return ast.literal_eval(s)
        except ValueError:
            return s

    def save(self, save_dir):
        pass

    def restore(self, save_dir):
        pass
