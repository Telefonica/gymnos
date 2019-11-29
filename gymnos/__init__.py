from . import models
from . import services
from . import datasets
from . import trackers
from . import preprocessors
from . import data_augmentors
from . import execution_environments

# MARK: Public API
from . import callbacks    # noqa: F401
from . import config    # noqa: F401
from . import trainer    # noqa: F401
from . import registration    # noqa: F401

# MARK: Datasets registration

datasets.register(
    type="boston_housing",
    entry_point="gymnos.datasets.boston_housing.BostonHousing"
)

datasets.register(
    type="data_usage_test",
    entry_point="gymnos.datasets.data_usage_test.DataUsageTest"
)

datasets.register(
    type="imdb",
    entry_point="gymnos.datasets.imdb.IMDB"
)

datasets.register(
    type="tiny_imagenet",
    entry_point="gymnos.datasets.tiny_imagenet.TinyImagenet"
)

datasets.register(
    type="dogs_vs_cats",
    entry_point="gymnos.datasets.dogs_vs_cats.DogsVsCats"
)

datasets.register(
    type="unusual_data_usage_test",
    entry_point="gymnos.datasets.unusual_data_usage_test.UnusualDataUsageTest"
)

datasets.register(
    type="synthetic_digits",
    entry_point="gymnos.datasets.synthetic_digits.SyntheticDigits"
)

datasets.register(
    type="mte",
    entry_point="gymnos.datasets.mte.MTE"
)

datasets.register(
    type="rock_paper_scissors",
    entry_point="gymnos.datasets.rock_paper_scissors.RockPaperScissors"
)

datasets.register(
    type="directory_image_classification",
    entry_point="gymnos.datasets.directory_image_classification.DirectoryImageClassification"
)

datasets.register(
    type="reddit_self_post_classification",
    entry_point="gymnos.datasets.reddit_self_post_classification.RedditSelfPostClassification"
)

datasets.register(
    type="hdf5",
    entry_point="gymnos.datasets.hdf5.HDF5Dataset"
)

datasets.register(
    type="titanic",
    entry_point="gymnos.datasets.titanic.Titanic"
)

# MARK: Services registration

services.register(
    type="http",
    entry_point="gymnos.services.http.HTTP"
)

services.register(
    type="kaggle",
    entry_point="gymnos.services.kaggle.Kaggle"
)

services.register(
    type="smb",
    entry_point="gymnos.services.smb.SMB"
)

services.register(
    type="sofia",
    entry_point="gymnos.services.sofia.SOFIA"
)

# MARK: Models registration

models.register(
    type="dogs_vs_cats_cnn",
    entry_point="gymnos.models.dogs_vs_cats_cnn.DogsVsCatsCNN"
)

models.register(
    type="data_usage_linear_regression",
    entry_point="gymnos.models.data_usage_linear_regression.DataUsageLinearRegression"
)

models.register(
    type="data_usage_holt_winters",
    entry_point="gymnos.models.data_usage_holt_winters.DataUsageHoltWinters"
)

models.register(
    type="unusual_data_usage_weighted_thresholds",
    entry_point="gymnos.models.unusual_data_usage_weighted_thresholds.UnusualDataUsageWT"
)

models.register(
    type="keras_classifier",
    entry_point="gymnos.models.keras.KerasClassifier"
)

models.register(
    type="keras_regressor",
    entry_point="gymnos.models.keras.KerasRegressor"
)

models.register(
    type="mte_nn",
    entry_point="gymnos.models.mte_nn.MTENN"
)

models.register(
type="repetition_ada_boost",
    entry_point="gymnos.models.repetition_ada_boost.RepetitionAdaBoost"
)

models.register(
    type="repetition_knn",
    entry_point="gymnos.models.repetition_knn.RepetitionKNN"
)

models.register(
    type="repetition_light_gbm",
    entry_point="gymnos.models.repetition_light_gbm.RepetitionLightGBM"
)

models.register(
    type="repetition_random_forest",
    entry_point="gymnos.models.repetition_random_forest.RepetitionRandomForest"
)

models.register(
    type="repetition_svm",
    entry_point="gymnos.models.repetition_svm.RepetitionSVM"
)

models.register(
    type="repetition_xgboost",
    entry_point="gymnos.models.repetition_xgboost.RepetitionXGBoost"
)

models.register(
    type="titanic",
    entry_point="gymnos.models.titanic.Titanic"
)

# MARK: Preprocessors registration

preprocessors.register(
    type="divide",
    entry_point="gymnos.preprocessors.divide.Divide"
)

preprocessors.register(
    type="grayscale",
    entry_point="gymnos.preprocessors.images.grayscale.Grayscale"
)

preprocessors.register(
    type="image_resize",
    entry_point="gymnos.preprocessors.images.image_resize.ImageResize"
)

preprocessors.register(
    type="replace",
    entry_point="gymnos.preprocessors.replace.Replace"
)

preprocessors.register(
    type="binary_vectorizer",
    entry_point="gymnos.preprocessors.texts.binary_vectorizer.BinaryVectorizer"
)

preprocessors.register(
    type="grayscale_to_color",
    entry_point="gymnos.preprocessors.images.grayscale_to_color.GrayscaleToColor"
)

preprocessors.register(
    type="lemmatization",
    entry_point="gymnos.preprocessors.texts.lemmatization.Lemmatization"
)

preprocessors.register(
    type="alphanumeric",
    entry_point="gymnos.preprocessors.texts.alphanumeric.Alphanumeric"
)

preprocessors.register(
    type="tfidf",
    entry_point="gymnos.preprocessors.texts.tfidf.Tfidf"
)

preprocessors.register(
    type="kbest",
    entry_point="gymnos.preprocessors.kbest.KBest"
)

preprocessors.register(
    type="standard_scaler",
    entry_point="gymnos.preprocessors.standard_scaler.StandardScaler"
)

preprocessors.register(
    type="utterances_aura_embeddings",
    entry_point="gymnos.preprocessors.utterances_aura_embeddings.UtterancesAuraEmbeddings"
)

preprocessors.register(
    type="utterances_embedding_pooling",
    entry_point="gymnos.preprocessors.utterances_embedding_pooling.UtterancesEmbeddingPooling"
)

# MARK: Tracker registration

trackers.register(
    type="comet_ml",
    entry_point="gymnos.trackers.comet_ml.CometML"
)

trackers.register(
    type="mlflow",
    entry_point="gymnos.trackers.mlflow.MLflow"
)

trackers.register(
    type="tensorboard",
    entry_point="gymnos.trackers.tensorboard.TensorBoard"
)

# MARK: Data Augmentors registration

data_augmentors.register(
    type="distort",
    entry_point="gymnos.data_augmentors.distort.Distort"
)

data_augmentors.register(
    type="flip",
    entry_point="gymnos.data_augmentors.flip.Flip"
)

data_augmentors.register(
    type="gaussian_distortion",
    entry_point="gymnos.data_augmentors.gaussian_distortion.GaussianDistortion"
)

data_augmentors.register(
    type="grayscale",
    entry_point="gymnos.data_augmentors.grayscale.Grayscale"
)

data_augmentors.register(
    type="histogram_equalisation",
    entry_point="gymnos.data_augmentors.histogram_equalisation.HistogramEqualisation"
)

data_augmentors.register(
    type="invert",
    entry_point="gymnos.data_augmentors.invert.Invert"
)

data_augmentors.register(
    type="random_brightness",
    entry_point="gymnos.data_augmentors.random_brightness.RandomBrightness"
)

data_augmentors.register(
    type="random_color",
    entry_point="gymnos.data_augmentors.random_color.RandomColor"
)

data_augmentors.register(
    type="random_contrast",
    entry_point="gymnos.data_augmentors.random_contrast.RandomContrast"
)

data_augmentors.register(
    type="random_erasing",
    entry_point="gymnos.data_augmentors.random_erasing.RandomErasing"
)

data_augmentors.register(
    type="rotate",
    entry_point="gymnos.data_augmentors.rotate.Rotate"
)

data_augmentors.register(
    type="rotate_range",
    entry_point="gymnos.data_augmentors.rotate_range.RotateRange"
)

data_augmentors.register(
    type="shear",
    entry_point="gymnos.data_augmentors.shear.Shear"
)

data_augmentors.register(
    type="skew",
    entry_point="gymnos.data_augmentors.skew.Skew"
)

data_augmentors.register(
    type="zoom",
    entry_point="gymnos.data_augmentors.zoom.Zoom"
)

data_augmentors.register(
    type="zoom_ground_truth",
    entry_point="gymnos.data_augmentors.zoom_ground_truth.ZoomGroundTruth"
)

data_augmentors.register(
    type="zoom_random",
    entry_point="gymnos.data_augmentors.zoom_random.ZoomRandom"
)

# MARK: Execution environments

execution_environments.register(
    type="fourth_platform",
    entry_point="gymnos.execution_environments.fourth_platform.FourthPlatform"
)
