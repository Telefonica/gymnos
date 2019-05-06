#
#
#   Spacy Utils
#
#

import spacy


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
        return spacy.load(language)
    except OSError:
        print("Downloading language model for the spaCy POS tagger\n"
              "(don't worry, this will only happen once)")
        from spacy.cli import download
        download(language)
        return spacy.load(language)
