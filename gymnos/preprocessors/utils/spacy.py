#
#
#   Spacy utils
#
#

from ...utils.lazy_imports import lazy_imports


def get_spacy_nlp(language="es"):
    """
    Get Spacy model for specified language. Download Spacy model if not found.

    Parameters
    ----------
    language: str
        Language to load or download

    Returns
    -------
    nlp: spacy.NLP
        Spacy model.
    """
    try:
        return lazy_imports.spacy.load(language)
    except OSError:
        print("Downloading language model for the spaCy POS tagger\n"
              "(don't worry, this will only happen once)")
        lazy_imports.spacy.cli.download(language)
        return lazy_imports.spacy.load(language)
