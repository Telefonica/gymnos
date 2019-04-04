#
#
#   Spacy Utils
#
#

import spacy


def get_spacy_nlp(language="es"):
    try:
        return spacy.load(language)
    except OSError:
        error  = "Model for language {lang} not found. "
        error += "Maybe you need to run python3 -m spacy download {lang}"
        raise OSError(error.format(lang=language))
