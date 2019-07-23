#
#
#   Loader
#
#


import os

from .utils.io_utils import import_from_json


def load(dataset=None, model=None, preprocessor=None, data_augmentor=None, tracker=None, **params):
    """
    Load dataset/model/preprocessor/data_augmentor/tracker instance from string specifying module id.

    Parameters
    ----------
    dataset: str, optional
        Dataset identifier.
    model: str, optional
        Model identifier
    preprocessor: str, optional
        Preprocessor identifier
    data_augmentor: str, optional
        Data augmentor identifier
    tracker: str, optional
        Tracker identifier.
    **params: any
        Parameters for module constructor

    Returns
    -------
    Model or Preprocessor or DataAugmentor or Tracker
    """
    if dataset is not None:
        return load_dataset(dataset, **params)
    elif model is not None:
        return load_model(model, **params)
    elif preprocessor is not None:
        return load_preprocessor(preprocessor, **params)
    elif data_augmentor is not None:
        return load_data_augmentor(data_augmentor, **params)
    elif tracker is not None:
        return load_tracker(tracker, **params)
    else:
        raise ValueError(("You must define one of the following parameters: dataset, model, preprocessor, "
                          "data_augmentor, tracker"))


def load_dataset(name, **params):
    try:
        Dataset = import_from_json(os.path.join(os.path.dirname(__file__), "var", "datasets.json"), name)
    except KeyError:
        raise ValueError("Dataset with name {} not found".format(name))
    return Dataset(**params)


def load_model(name, **params):
    try:
        Model = import_from_json(os.path.join(os.path.dirname(__file__), "var", "models.json"), name)
    except KeyError:
        raise ValueError("Model with name {} not found".format(name))
    return Model(**params)


def load_preprocessor(name, **params):
    try:
        Preprocessor = import_from_json(os.path.join(os.path.dirname(__file__), "var", "preprocessors.json"), name)
    except KeyError:
        raise ValueError("Preprocessor with name {} not found".format(name))
    return Preprocessor(**params)


def load_data_augmentor(name, **params):
    try:
        DataAugmentor = import_from_json(os.path.join(os.path.dirname(__file__), "var", "data_augmentors.json"), name)
    except KeyError:
        raise ValueError("Data augmentor with name {} not found".format(name))
    return DataAugmentor(**params)


def load_tracker(name, **params):
    try:
        Tracker = import_from_json(os.path.join(os.path.dirname(__file__), "var", "trackers.json"), name)
    except KeyError:
        raise ValueError("Tracker with name {} not found".format(name))
    return Tracker(**params)
