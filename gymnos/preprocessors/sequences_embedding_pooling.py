#
#
#   Sequences Embeddings Pooling
#
#

import numpy as np

from .preprocessor import Preprocessor


class SequencesEmbeddingPooling(Preprocessor):
    """
    The SequencesEmbeddingPooling class consists of a series of pooling in which different statistics are
    concatenated to a new component of the embedding array.

    These pooling are made two levels (phrase and sequence) being the sequence pooling constructed
    from the phrase pooling.

    The combinations used in the proof of concept are (first step to phrase and then to sequence):

        - 'median_and_flatten_sequence',
        - 'median_max_and_flatten_sequence',
        - 'median_max_min_and_flatten_sequence',

        - 'mean_and_flatten_sequence',
        - 'mean_max_and_flatten_sequence',
        - 'mean_max_min_and_flatten_sequence',

        - 'median_and_median',
        - 'median_and_median_max',
        - 'median_and_median_max_min',

        - 'median_max_and_median',
        - 'median_max_and_median_max',
        - 'median_max_and_median_max_min',

        - 'median_max_min_and_median',
        - 'median_max_min_and_median_max',
        - 'median_max_min_and_median_max_min',

        - 'mean_and_median',
        - 'mean_and_median_max',
        - 'mean_and_median_max_min',

        - 'mean_max_and_median',
        - 'mean_max_and_median_max',
        - 'mean_max_and_median_max_min',


        - 'mean_max_min_and_median',
        - 'mean_max_min_and_median_max',
        - 'mean_max_min_and_median_max_min'


    Parameters
    -----------
    type_pooling: str,
                Type of pooling
    """

    def __init__(self, type_pooling="flatten_sequence"):
        poolings = ['median_and_flatten_sequence',
                    'max_median_and_flatten_sequence',
                    'max_median_min_and_flatten_sequence',
                    'mean_and_flatten_sequence',
                    'mean_max_and_flatten_sequence',
                    'mean_max_min_and_flatten_sequence',
                    'median_and_median',
                    'median_and_max_median',
                    'median_and_max_median_min',
                    'max_median_and_mean_max',
                    'max_median_and_median',
                    'max_median_and_max_median',
                    'max_median_and_mean_max_min',
                    'max_median_and_max_median_min',
                    'max_median_min_and_median',
                    'max_median_min_and_max_median',
                    'max_median_min_and_max_median_min',
                    'mean_and_median',
                    'mean_and_max_median',
                    'mean_and_max_median_min',
                    'mean_max_and_median',
                    'mean_max_and_max_median',
                    'mean_max_and_max_median_min',
                    'mean_max_min_and_median',
                    'mean_max_min_and_max_median',
                    'mean_max_min_and_max_median_min']

        try:
            assert type_pooling in poolings
        except AssertionError as e:
            e.args += (type_pooling, "is not in list of valid poolings values:", poolings)
            raise
        self.type_pooling = type_pooling

    def fit(self, x, y=None):
        return self

    def fit_generator(self, generator):
        return self

    def transform(self, x):
        type_pooling_sequence, type_pooling_phrase = self.type_pooling.split("_and_")
        x = SequencesEmbeddingPooling.__apply_pooling(element=x, type_pooling_sequence=type_pooling_sequence,
                                                      type_pooling_phrase=type_pooling_phrase)
        return x

    @staticmethod
    def __apply_pooling(element, type_pooling_sequence, type_pooling_phrase):
        """
        Applies a pooling to a list of arrays concatenating several statistics to a sequences and phrases depending on
        type_pooling parameters.

        Parameters
        -----------
        element: array,
            input data.
        type_pooling_sequence: str
            type of pooling applied to sequences.
         type_pooling_phrase: str
            type of pooling applied to phrases.
        """
        total_sequence = []
        for sequence in list(element):
            total_phrase = []
            for phrase in sequence:
                if type_pooling_phrase == "flatten_sequence":
                    total_phrase = [item for sublist in sequence for item in sublist]
                else:
                    total_phrase.append(
                        SequencesEmbeddingPooling.__pooling(element=phrase, type_pooling=type_pooling_phrase))
            total_sequence.append(
                SequencesEmbeddingPooling.__pooling(element=total_phrase, type_pooling=type_pooling_sequence))
        return total_sequence

    @staticmethod
    def __pooling(element, type_pooling):
        """
        Makes a pooling to a list of arrays concatenating several statistics depending on type_pooling parameter.

        Parameters
        -----------
        element: , list
            list of arrays.Input data.
        type_pooling: str

        """
        result = None

        if type_pooling == "median":
            result = np.median(element, axis=0)
        elif type_pooling == "max_median":
            result = np.concatenate((np.max(element, axis=0), np.median(element, axis=0)))
        elif type_pooling == "median_min":
            result = np.concatenate((np.median(element, axis=0), np.min(element, axis=0)))
        elif type_pooling == "max_median_min":
            result = np.concatenate((np.max(element, axis=0), np.median(element, axis=0), np.min(element, axis=0)))
        elif type_pooling == "mean":
            result = np.mean(element, axis=0)
        elif type_pooling == "mean_max":
            result = np.concatenate((np.mean(element, axis=0), np.max(element, axis=0)))
        elif type_pooling == "mean_max_min":
            result = np.concatenate((np.mean(element, axis=0), np.max(element, axis=0), np.min(element, axis=0)))
        else:
            pass
        return result

    def save(self, save_dir):
        pass

    def restore(self, save_dir):
        pass
