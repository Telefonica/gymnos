#
#
#   Dataset
#
#

import logging

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
    name: str
        Name of dataset, the current available datasets are the following:

        The current available datasets are the following:

        - ``"boston_housing"``: :class:`lib.datasets.boston_housing.BostonHousing`,
        - ``"dogs_vs_cats"``: :class:`lib.datasets.dogs_vs_cats.DogsVsCats`,
        - ``"imdb"``: :class:`lib.datasets.imdb.IMDB`,
        - ``"tiny_imagenet"``: :class:`lib.datasets.tiny_imagenet.TinyImagenet`,
        - ``"synthetic_digits"``: :class:`lib.datasets.synthetic_digits.SyntheticDigits`,
        - ``"mte"``: :class:`lib.datasets.mte.MTE`,
        - ``"data_usage_test"``: :class:`lib.datasets.data_usage_test.DataUsageTest`,
        - ``"unusual_data_usage_test"``: :class:`lib.datasets.unusual_data_usage_test.UnusualDataUsageTest`
        - ``"rock_paper_scissors"``: :class:`lib.datasets.rock_paper_scissors.RockPaperScissors`

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

        The current available preprocessors are the following:

        - ``"divide"``: :class:`lib.preprocessors.divide.Divide`,
        - ``"grayscale"``: :class:`lib.preprocessors.images.grayscale.Grayscale`,
        - ``"image_resize"``: :class:`lib.preprocessors.images.image_resize.ImageResize`,
        - ``"replace"``: :class:`lib.preprocessors.replace.Replace`,
        - ``"grayscale_to_color"``: :class:`lib.preprocessors.images.grayscale_to_color.GrayscaleToColor`,
        - ``"lemmatization"``: :class:`lib.preprocessors.texts.lemmatization.Lemmatization`,
        - ``"alphanumeric"``: :class:`lib.preprocessors.texts.alphanumeric.Alphanumeric`,
        - ``"tfidf"``: :class:`lib.preprocessors.texts.tfidf.Tfidf`,
        - ``"kbest"``: :class:`lib.preprocessors.kbest.KBest`,
        - ``"standard_scaler"``: `sklearn.preprocessing.StandardScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_ # noqa: E501
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
        The current available preprocessors are the following:
            TODO

    Examples
    --------
    .. code-block:: py

        Dataset(
            name= "tiny_imagenet",
            one_hot=True,
            samples={
                "train": 0.1,
                "test": 0.1
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

    def __init__(self, name, samples=None, preprocessors=None, seed=None, shuffle=True, one_hot=False, chunk_size=None,
                 data_augmentors=None):
        samples = samples or {}
        preprocessors = preprocessors or []
        data_augmentors = data_augmentors or []

        self.name = name
        self.seed = seed
        self.one_hot = one_hot
        self.shuffle = shuffle
        self.chunk_size = chunk_size

        self.samples = DatasetSamples(**samples)

        self.dataset = datasets.load(name)

        # we save these specs so we can export it via to_dict
        self.preprocessors_specs = preprocessors
        self.data_augmentors_specs = data_augmentors

        self.preprocessors = PreprocessorPipeline.from_dict(preprocessors)
        self.data_augmentors = DataAugmentorPipeline.from_dict(data_augmentors)

    def to_dict(self):
        return dict(
            name=self.name,
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
