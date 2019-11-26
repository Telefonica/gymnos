#
#
#   Preprocessor
#
#

from . import load

from tqdm import tqdm
from copy import deepcopy
from abc import ABCMeta, abstractmethod

from ..utils.py_utils import drop


class Preprocessor(metaclass=ABCMeta):
    """
    Base class for all Gymnos preprocessors.

    You need to implement the following methods: ``fit`` and optionally ``transform``.
    """

    @abstractmethod
    def fit(self, X, y=None):
        """
        Fit preprocessor to training data.

        Parameters
        ----------
        X: array_like
            Features
        y: array_like, optional
            Labels

        Returns
        --------
        self: Preprocessor
            Own instance for chain purposes.
        """

    def fit_generator(self, generator):
        """
        Fit preprocessors to generator.

        Parameters
        ----------
        generator: generator
            Generator iterating X and y

        Returns
        -------
        self: Preprocessor
            Own instance for chain purposes.
        """
        raise NotImplementedError(("Preprocessor {} don't implement fit_generator "
                                   "method").format(self.__class__.__name__))

    @abstractmethod
    def transform(self, X):
        """
        Transform data.

        Parameters
        ----------
        X: array_like
            Features

        Returns
        -------
        X_t: array_like
            Transformed features
        """


class SparkPreprocessor(metaclass=ABCMeta):
    """
    Base class for all Gymnos Spark preprocessors.
    You need to implement the following methods: fit, transform.

    Parameters
    ------------
    features_col: str
        Column name for your features
    outputs_col: str
        Column name to output results
    labels_col: str
        Column name for your labels
    """

    def __init__(self, features_col, outputs_col="outputs", labels_col="labels"):
        self.features_col = features_col
        self.outputs_col = outputs_col
        self.labels_col = labels_col

    @abstractmethod
    def fit(self, dataset):
        """
        Fit preprocessor to Spark DataFrame

        Parameters
        -----------
        dataset: pyspark.sql.DataFrame
            Dataset to fit.
        """

    @abstractmethod
    def transform(self, dataset):
        """
        Transform Spark DataFrame

        Parameters
        -----------
        dataset: pyspark.sql.DataFrame
            Transform Spark DataFrame. Transformed column will be``self.outputs_col``.
        """

class Pipeline:
    """
    Pipeline to chain preprocessors.

    Parameters
    -----------
    preprocessors: list of Preprocessor or list SparkPreprocessor, optional
        Preprocessors queue.
    """

    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors or []

    @staticmethod
    def from_dict(specs):
        from . import load

        preprocessors = []
        for preprocessor_spec in specs:
            preprocessor_spec = deepcopy(preprocessor_spec)
            preprocessor_type = preprocessor_spec["type"]
            preprocessor = load(preprocessor_type, **drop(preprocessor_spec, "type"))
            preprocessors.append(preprocessor)

        return Pipeline(preprocessors)

    def reset(self):
        """
        Empty preprocessors queue.
        """
        self.preprocessors = []

    def add(self, preprocessor):
        """
        Parameters
        -----------
        preprocessor: Preprocessor or SparkPreprocessor
            Preprocessor to append to queue.
        """
        self.preprocessors.append(preprocessor)

    def fit(self, *args):
        """
        Parameters
        -----------
        *args: any
            Arguments for Preprocessor.fit or SparkPreprocessor.fit
        """
        if not self.preprocessors:
            return self

        features = args[0]
        pbar = tqdm(self.preprocessors)

        for preprocessor in pbar:
            pbar.set_description("Fitting with {}".format(preprocessor.__class__.__name__))
            if len(args) > 1:
                preprocessor.fit(features, args[1])
            else:
                preprocessor.fit(features)

            features = preprocessor.transform(features)

        return self

    def fit_generator(self, generator):
        """
        Parameters
        -----------
        generator: Iterable
            Iterable yielding features and labels
        """
        if not self.preprocessors:
            return self

        fitted_preprocessors = []
        pbar = tqdm(self.preprocessors)
        for preprocessor in pbar:
            def generator_with_transform():
                for X, y in generator:
                    for fitted_preprocessor in fitted_preprocessors:
                        X = fitted_preprocessor.transform(X)
                    yield X, y

            pbar.set_description("Fitting generator with {}".format(preprocessor.__class__.__name__))
            preprocessor.fit_generator(generator_with_transform())
            fitted_preprocessors.append(preprocessor)

        return self

    def transform(self, dataset):
        """
        Parameters
        -----------
        dataset: np.ndarray, pd.DataFrame or pyspark.sql.DataFrame
            Dataset to transform
        """
        if not self.preprocessors:
            return dataset

        for preprocessor in self.preprocessors:
            dataset = preprocessor.transform(dataset)

        return dataset

    def __str__(self):
        preprocessors_names = [prep.__class__.__name__ for prep in self.preprocessors]
        return " | ".join(preprocessors_names)

    def __len__(self):
        """
        Returns number of preprocessors
        """
        return len(self.preprocessors)


