#
#
#   Dataset
#
#

import os

from ..utils.io_utils import import_from_json
from ..preprocessors import Pipeline
from ..logger import get_logger

DATASETS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "datasets.json")
PREPROCESSORS_IDS_TO_MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "var", "preprocessors.json")


class DatasetSamples:

    def __init__(self, train=None, test=None):
        if train is None and test is None:
            test = 0.25
            train = 1 - test
        elif train is None:
            train = 1 - test
        elif test is None:
            test = 1 - train

        self.train = train
        self.test = test

        if (self.test + self.train < 1.0):
            get_logger(prefix=self).warning("Using only {:.2f}% of total data".format(self.train + self.test))


class Dataset:
    """
    Parameters
    ----------
    name: str
        Name of dataset, the current available datasets are the following:

        The current available datasets are the following:

        - ``"boston_housing"``: :class:`lib.datasets.boston_housing.BostonHousing`,
        - ``"cifar10"``: :class:`lib.datasets.cifar10.CIFAR10`,
        - ``"dogs_vs_cats"``: :class:`lib.datasets.dogs_vs_cats.DogsVsCats`,
        - ``"fashion_mnist"``: :class:`lib.datasets.fashion_mnist.FashionMNIST`,
        - ``"imdb"``: :class:`lib.datasets.imdb.IMDB`,
        - ``"kddcup99"``: :class:`lib.datasets.kddcup99.KDDCup99`,
        - ``"mnist"``: :class:`lib.datasets.mnist.MNIST`,
        - ``"tiny_imagenet"``: :class:`lib.datasets.tiny_imagenet.TinyImagenet`,
        - ``"mte"``: :class:`lib.datasets.mte.MTE`,
        - ``"data_usage_test"``: :class:`lib.datasets.data_usage_test.DataUsageTest`,
        - ``"unusual_data_usage_test"``: :class:`lib.datasets.unusual_data_usage_test.UnusualDataUsageTest`

    samples: dict, optional
        Samples to split dataset into random train and test subsets

        The following properties are available:

        **train**: `float` or `int`, optional
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to
            include in the train split.
            If int, represents the absolute number of train samples.
            If None, the value is automatically set to the complement of the test size.
        **test**: `float` or `int`, optional
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to
            include in the test split.
            If int, represents the absolute number of test samples.
            If None, the value is set to the complement of the train size.
            If None and ``train_size`` is unspecified, by default the value is set to 0.25.
    preprocessors: list of dict, optional
        List of preprocessors to apply to dataset. This property requires a list with dictionnaries with at least
        a ``type`` field specifying the type of preprocessor.  The other properties are the properties for that
        preprocessor.

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
        - ``"standard_scaler"``: `sklearn.preprocessing.StandardScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_
    seed: int, optional
        Seed used to split dataset.
    shuffle: bool, optional
        Whether or not to shuffle the data before splitting.
    one_hot: bool, optional
        Whether or not to one-hot encode labels. Required for some models.

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
    """

    def __init__(self, name, samples=None, preprocessors=None, seed=None, shuffle=True, one_hot=False, chunk_size=None):
        samples = samples or {}
        preprocessors = preprocessors or []

        self.name = name
        self.seed = seed
        self.one_hot = one_hot
        self.shuffle = shuffle
        self.chunk_size = chunk_size

        self.samples = DatasetSamples(**samples)

        DatasetClass = import_from_json(DATASETS_IDS_TO_MODULES_PATH, name)
        self.dataset = DatasetClass()

        self.pipeline = Pipeline()
        for preprocessor_config in preprocessors:
            PreprocessorClass = import_from_json(PREPROCESSORS_IDS_TO_MODULES_PATH, preprocessor_config.pop("type"))
            preprocessor = PreprocessorClass(**preprocessor_config)
            self.pipeline.add(preprocessor)
