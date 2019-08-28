#
#
#   Binary Vectorizer
#
#

from ..preprocessor import Preprocessor

from tensorflow.keras.preprocessing.text import Tokenizer


class BinaryVectorizer(Preprocessor):
    """
    Vectorize a text corpus by turning each text into a one-hot vector indicating whether or not each word
    is present in the document.

    Parameters
    ----------
    num_words: int, optional
        The maximum number of words to keep, based on word frequency.
        Only the most common ``num_words-1`` words will be kept.
    filters: str, optional
        A string where each element is a character that will be filtered from the texts.
        The default is all punctuation, plus tabs and line breaks, minus the ' character.
    lower: bool, optional
        Whether to convert the texts to lowercase.
    split: str, optional
        Separator for word splitting.
    """

    def __init__(self, num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '):
        self.tokenizer = Tokenizer(num_words=num_words, filters=filters, lower=lower, split=split)

    def fit(self, X, y=None):
        self.tokenizer.fit_on_texts(X)
        return self

    def transform(self, X):
        return self.tokenizer.texts_to_matrix(X, mode="binary")
