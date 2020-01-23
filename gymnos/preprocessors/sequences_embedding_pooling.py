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
    concatenated to a new component of the embedding array. These pooling are made two levels (phrase and sequence)
    being the sequence pooling constructed from the phrase pooling.

    The combinations used in the proof of concept are:

    'two_steps_flatten_sequence_and_mean_max_min',
    'two_steps_flatten_sequence_and_max_median',
    'two_steps_mean_and_mean_max_min',
    'two_steps_mean_max_and_mean_max_min',
    'two_steps_max_median_and_mean_max_min',
    'two_steps_mean_max_min_and_mean_max_min',
    'two_steps_median_and_mean_max_min',
    'two_steps_max_median_and_max_median',
    'two_steps_mean_and_max_median',
    'two_steps_mean_max_and_max_median',
    'two_steps_max_median_and_max_median',
    'two_steps_mean_max_min_and_max_median'

    Parameters
    -----------
    type_pooling: str,
                Type of pooling
    """

    def __init__(self, type_pooling="flatten_sequence"):
        poolings = ['two_steps_flatten_sequence_and_mean_max_min',
                    'two_steps_flatten_sequence_and_max_median',
                    'two_steps_mean_and_mean_max_min',
                    'two_steps_mean_max_and_mean_max_min',
                    'two_steps_max_median_and_mean_max_min',
                    'two_steps_mean_max_min_and_mean_max_min',
                    'two_steps_median_and_mean_max_min',
                    'two_steps_max_median_and_max_median',
                    'two_steps_mean_and_max_median',
                    'two_steps_mean_max_and_max_median',
                    'two_steps_max_median_and_max_median',
                    'two_steps_mean_max_min_and_max_median']
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
        if self.type_pooling == "two_steps_flatten_sequence_and_mean_max_min":
            x = SequencesEmbeddingPooling.__apply_pooling(list_of_list=x, type_pooling_sequence="mean_max_min",
                                                          type_pooling_phrase="flatten_sequence")
        if self.type_pooling == "two_steps_flatten_sequence_and_max_median":
            x = SequencesEmbeddingPooling.__apply_pooling(list_of_list=x, type_pooling_sequence="max_median",
                                                          type_pooling_phrase="flatten_sequence")
        elif self.type_pooling == "two_steps_mean_and_mean_max_min":
            x = SequencesEmbeddingPooling.__apply_pooling(list_of_list=x, type_pooling_sequence="mean_max_min",
                                                          type_pooling_phrase="mean")
        elif self.type_pooling == "two_steps_mean_max_and_mean_max_min":
            x = SequencesEmbeddingPooling.__apply_pooling(list_of_list=x, type_pooling_sequence="mean_max_min",
                                                          type_pooling_phrase="mean_max")
        elif self.type_pooling == "two_steps_max_median_and_mean_max_min":
            x = SequencesEmbeddingPooling.__apply_pooling(list_of_list=x, type_pooling_sequence="mean_max_min",
                                                          type_pooling_phrase="max_median")
        elif self.type_pooling == "two_steps_mean_max_min_and_mean_max_min":
            x = SequencesEmbeddingPooling.__apply_pooling(list_of_list=x, type_pooling_sequence="mean_max_min",
                                                          type_pooling_phrase="mean_max_min")
        elif self.type_pooling == "two_steps_median_and_mean_max_min":
            x = SequencesEmbeddingPooling.__apply_pooling(list_of_list=x, type_pooling_sequence="mean_max_min",
                                                          type_pooling_phrase="median")
        if self.type_pooling == "two_steps_max_median_and_max_median":
            x = SequencesEmbeddingPooling.__apply_pooling(list_of_list=x, type_pooling_sequence="max_median",
                                                          type_pooling_phrase="max_median")
        elif self.type_pooling == "two_steps_mean_and_max_median":
            x = SequencesEmbeddingPooling.__apply_pooling(list_of_list=x, type_pooling_sequence="max_median",
                                                          type_pooling_phrase="mean")
        elif self.type_pooling == "two_steps_mean_max_and_max_median":
            x = SequencesEmbeddingPooling.__apply_pooling(list_of_list=x, type_pooling_sequence="max_median",
                                                          type_pooling_phrase="mean_max")
        elif self.type_pooling == "two_steps_max_median_and_max_median":
            x = SequencesEmbeddingPooling.__apply_pooling(list_of_list=x, type_pooling_sequence="max_median",
                                                          type_pooling_phrase="max_median")
        elif self.type_pooling == "two_steps_mean_max_min_and_max_median":
            x = SequencesEmbeddingPooling.__apply_pooling(list_of_list=x, type_pooling_sequence="max_median",
                                                          type_pooling_phrase="mean_max_min")
        else:
            pass
        return x

    @staticmethod
    def __apply_pooling(list_of_list, type_pooling_sequence, type_pooling_phrase):
        """
        Applies a pooling to a list of arrays concatenating several statistics to a sequences and phrases depending on
        type_pooling parameters.

        Parameters
        -----------
        list_of_list: list,
            list of arrays.Input data.
        type_pooling_sequence: str
            type of pooling applied to sequences.
         type_pooling_phrase: str
            type of pooling applied to phrases.
        """
        total_sequence = []
        for sequence in list_of_list:
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
        if type_pooling == "mean":
            result = np.mean(element, axis=0)
        elif type_pooling == "median":
            result = np.median(element, axis=0)
        elif type_pooling == "mean_max":
            result = np.concatenate((np.mean(element, axis=0), np.max(element, axis=0)))
        elif type_pooling == "max_median":
            result = np.concatenate((np.max(element, axis=0), np.median(element, axis=0)))
        elif type_pooling == "min_median":
            result = np.concatenate((np.min(element, axis=0), np.median(element, axis=0)))
        elif type_pooling == "mean_median":
            result = np.concatenate((np.mean(element, axis=0), np.median(element, axis=0)))
        elif type_pooling == "mean_max_median":
            result = np.concatenate((np.mean(element, axis=0), np.max(element, axis=0), np.median(element, axis=0)))
        elif type_pooling == "mean_max_min_median":
            result = np.concatenate((np.mean(element, axis=0), np.max(element, axis=0), np.min(element, axis=0),
                                     np.median(element, axis=0)))
        elif type_pooling == "mean_min":
            result = np.concatenate((np.mean(element, axis=0), np.min(element, axis=0)))
        elif type_pooling == "mean_max_min":
            result = np.concatenate((np.mean(element, axis=0), np.max(element, axis=0), np.min(element, axis=0)))
        elif type_pooling == "mean":
            result = np.mean(element, axis=0)
        elif type_pooling == "max":
            result = np.max(element, axis=0)
        elif type_pooling == "min":
            result = np.min(element, axis=0)
        else:
            pass
        return result

    def save(self, save_dir):
        pass

    def restore(self, save_dir):
        pass
