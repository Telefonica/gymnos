#
#
#   Utterances Embeddings Pooling
#
#

import numpy as np

from gymnos.preprocessors.preprocessor import Preprocessor


class UtterancesEmbeddingPooling(Preprocessor):
    """
    This class applies pooling to list of list of arrays concatenating several combinations of statistics.

    Parameters
    -----------
    type_pooling: str,
                Type of pooling (flatten_sequence, two_steps_complete_mean, two_steps_complete_mean_max
                or two_steps_complete_complete)
    """

    def __init__(self, type_pooling):
        self.type_pooling = type_pooling

    def fit(self, X, y=None):
        return self

    def fit_generator(self, generator):
        return self

    def transform(self, X):
        if self.type_pooling == "flatten_sequence":
            X = UtterancesEmbeddingPooling.__apply_pooling(list_of_list=X, type_pooling_sequence="complete",
                                                           type_pooling_phrase="flatten_sequence")
        elif self.type_pooling == "two_steps_complete_mean":
            X = UtterancesEmbeddingPooling.__apply_pooling(list_of_list=X, type_pooling_sequence="complete",
                                                           type_pooling_phrase="mean")
        elif self.type_pooling == "two_steps_complete_mean_max":
            X = UtterancesEmbeddingPooling.__apply_pooling(list_of_list=X, type_pooling_sequence="complete",
                                                           type_pooling_phrase="mean_max")
        elif self.type_pooling == "two_steps_complete_complete":
            X = UtterancesEmbeddingPooling.__apply_pooling(list_of_list=X, type_pooling_sequence="complete",
                                                           type_pooling_phrase="complete")
        return X

    @staticmethod
    def __apply_pooling(list_of_list, type_pooling_sequence, type_pooling_phrase):
        """
        Applies a pooling to a list of arrays concatenating diferent statitics to a seneces and phrases depending on
        type_pooling parameters.

        Parameters
        -----------
        list_of_list: list,
            list of arrays.Input data.
        type_pooling_sequence: str
            type of pooling applied to sequences(mean,mean_max_complete)
         type_pooling_phrase: str
            type of pooling applied to phrases(mean,mean_max_complete)
        """
        total_sequence = []
        for sequence in list_of_list:
            total_phrase = []
            for phrase in sequence:
                if type_pooling_phrase == "flatten_sequence":
                    total_phrase = [item for sublist in sequence for item in sublist]
                else:
                    total_phrase.append(
                        UtterancesEmbeddingPooling.pooling(element=phrase, type_pooling=type_pooling_phrase))
            total_sequence.append(
                UtterancesEmbeddingPooling.pooling(element=total_phrase, type_pooling=type_pooling_sequence))
        return total_sequence

    @staticmethod
    def pooling(element, type_pooling):
        """
        Makes a pooling to a list of arrays concatenating diferent statitics depending on type_pooling parameter.

        Parameters
        -----------
        element: , list
            list of arrays.Input data.
        type_pooling: str
            type of pooling applied.
            if type_pooling=mean, concatenates: mean.
            if type_pooling=mean_max, concatenates: mean and max.
            if type_pooling=complete, concatenates: mean and max and min.
        """
        result = None
        if type_pooling == "mean":
            result = np.mean(element, axis=0)
        elif type_pooling == "mean_max":
            result = np.concatenate((np.mean(element, axis=0), np.max(element, axis=0)))
        elif type_pooling == "complete":
            result = np.concatenate((np.mean(element, axis=0), np.max(element, axis=0), np.min(element, axis=0)))
        return result

    def save(self, save_dir):
        pass

    def restore(self, save_dir):
        pass
