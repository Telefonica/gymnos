#
#
#   TFIDF Transformer
#
#

from sklearn.feature_extraction.text import TfidfVectorizer

from ..utils.spacy import get_spacy_nlp
from ...utils.lazy_imports import lazy_imports
from ..preprocessor import Preprocessor


class Tfidf(Preprocessor):
    """
    Compute term frequency–inverse document frequency.

    Note
    ----
    Refer to `sklearn.feature_extraction.text.TfidfVectorizer <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>`_
    for parameter details.
    """ # noqa: E501

    def __init__(self, lowercase=True, strip_accents=None, language="english", skip_stop_words=True, ngram_range=(1, 1),
                 max_df=1.0, min_df=1.0, max_features=None, use_idf=True, sublinear_tf=False):

        improved_tokenizer = self.improve_tokenizer_if_possible(language)

        if skip_stop_words:
            stop_words = None
        else:
            stop_words = self.stop_words_for_language(language)

        self.tfidf = TfidfVectorizer(
            lowercase=lowercase,
            strip_accents=strip_accents,
            stop_words=stop_words,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            use_idf=use_idf,
            sublinear_tf=sublinear_tf,
            tokenizer=improved_tokenizer
        )

    def improve_tokenizer_if_possible(self, language):
        if language == "english":
            nlp = get_spacy_nlp("en")
        elif language == "spanish":
            nlp = get_spacy_nlp("es")
        else:
            return None

        def spacy_tokenizer(document):
            doc = nlp(document)
            return [token.text for token in doc]

        return spacy_tokenizer

    def stop_words_for_language(self, language):
        if language == "english":
            return lazy_imports.spacy.lang.en.stop_words.STOP_WORDS
        if language == "spanish":
            return lazy_imports.spacy.lang.es.stop_words.STOP_WORDS
        else:
            return None

    def fit(self, X, y=None):
        self.tfidf.fit(X, y)
        return self

    def transform(self, X):
        X_t = self.tfidf.transform(X)
        return X_t.toarray()  # convert sparse to dense