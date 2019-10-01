#
#
#   Dataset
#
#

import logging

from copy import deepcopy

from .. import datasets
from ..preprocessors.preprocessor import Pipeline as PreprocessorPipeline
from ..data_augmentors.data_augmentor import Pipeline as DataAugmentorPipeline

logger = logging.getLogger(__name__)


class DatasetSamples:

    def __init__(self, train=None, test=None):
        if (isinstance(train, float)) and isinstance(test, float) and (train + test > 1.0):
            raise ValueError("If train/test datatype is float, it must be lower than 1.0")

        if train is None and test is None:
            train = 0.75
            test = 1 - train
        elif train is None:
            train = 1 - test
        elif test is None:
            test = 1 - train

        self.test = test
        self.train = train

        if (self.test + self.train < 1.0):
            logger.warning("Using only {:.2f}% of total data".format(self.train + self.test))


class Dataset:
    """
    Parameters
    ----------
    dataset: dict
        Dataset type and their parameters with the structure ``{"type", **parameters}``
    samples: dict, optional
        Samples to split dataset into random train and test subsets

        The following properties are available:

        **train**: `float` or `int`, optional
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
            If int, represents the absolute number of train samples.
            If None, the value is automatically set to the complement of the test size.
        **test**: `float` or `int`, optional
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
            If int, represents the absolute number of test samples.
            If None, the value is set to the complement of the train size.
            If None and ``train_size`` is unspecified, by default the value is set to 0.25.
    preprocessors: list of dict, optional
        List of preprocessors to apply to dataset. This property requires a list with dictionnaries with at least
        a ``type`` field specifying the type of preprocessor.  The other properties are the properties for that preprocessor.
    seed: int, optional
        Seed used to split dataset.
    shuffle: bool, optional
        Whether or not to shuffle the data before splitting.
    one_hot: bool, optional
        Whether or not to one-hot encode labels. Required for some models.
    data_augmentors: list of dict, optional
        List of data augmentors to apply to images dataset. This property requires a list with dictionnaries with at
        least a ``type`` field specifying the type of data augmentor.  The other properties are the properties for that
        data augmentor.

    Examples
    --------
    .. code-block:: py

        Dataset(
            dataset={
                "type": "tiny_imagenet"
            },
            one_hot=True,
            samples={
                "train": 0.8,
                "test": 0.2
            },
            preprocessors=[
                {
                    "type": "grayscale"
                },
                {
                    "type": "image_resize",
                    "width": 25,
                    "height": 25
                }
            ]
        )
    """  # noqa: E501

    def __init__(self, dataset, samples=None, preprocessors=None, seed=None, shuffle=True, one_hot=False,
                 chunk_size=None, data_augmentors=None):
        samples = samples or {}
        preprocessors = preprocessors or []
        data_augmentors = data_augmentors or []

        self.seed = seed
        self.one_hot = one_hot
        self.shuffle = shuffle
        self.chunk_size = chunk_size

        self.samples = DatasetSamples(**samples)

        self.dataset_spec = deepcopy(dataset)

        self.dataset = datasets.load(**dataset)

        # we save these specs so we can export it via to_dict
        self.preprocessors_specs = deepcopy(preprocessors)
        self.data_augmentors_specs = deepcopy(data_augmentors)

        self.preprocessors = PreprocessorPipeline.from_dict(preprocessors)
        self.data_augmentors = DataAugmentorPipeline.from_dict(data_augmentors)

    def to_dict(self):
        return dict(
            dataset=self.dataset_spec,
            seed=self.seed,
            one_hot=self.one_hot,
            shuffle=self.shuffle,
            chunk_size=self.chunk_size,
            samples=dict(
                train=self.samples.train,
                test=self.samples.test
            ),
            preprocessors=self.preprocessors_specs,
            data_augmentors=self.data_augmentors_specs
        )
