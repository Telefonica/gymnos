from . import models
from . import services
from . import datasets
from . import trackers
from . import preprocessors
from . import data_augmentors


# MARK: Datasets registration

datasets.register(
    name="boston_housing",
    entry_point="gymnos.datasets.boston_housing.BostonHousing"
)

datasets.register(
    name="data_usage_test",
    entry_point="gymnos.datasets.data_usage_test.DataUsageTest"
)

datasets.register(
    name="imdb",
    entry_point="gymnos.datasets.imdb.IMDB"
)

datasets.register(
    name="tiny_imagenet",
    entry_point="gymnos.datasets.tiny_imagenet.TinyImagenet"
)

datasets.register(
    name="dogs_vs_cats",
    entry_point="gymnos.datasets.dogs_vs_cats.DogsVsCats"
)

datasets.register(
    name="unusual_data_usage_test",
    entry_point="gymnos.datasets.unusual_data_usage_test.UnusualDataUsageTest"
)

datasets.register(
    name="synthetic_digits",
    entry_point="gymnos.datasets.synthetic_digits.SyntheticDigits"
)

datasets.register(
    name="mte",
    entry_point="gymnos.datasets.mte.MTE"
)

datasets.register(
    name="rock_paper_scissors",
    entry_point="gymnos.datasets.rock_paper_scissors.RockPaperScissors"
)

# MARK: Services registration

services.register(
    name="http",
    entry_point="gymnos.services.http.HTTP"
)

services.register(
    name="kaggle",
    entry_point="gymnos.services.kaggle.Kaggle"
)

services.register(
    name="smb",
    entry_point="gymnos.services.smb.SMB"
)

# MARK: Models registration

models.register(
    name="dogs_vs_cats_cnn",
    entry_point="gymnos.models.dogs_vs_cats_cnn.DogsVsCatsCNN"
)

models.register(
    name="data_usage_linear_regression",
    entry_point="gymnos.models.data_usage_linear_regression.DataUsageLinearRegression"
)

models.register(
    name="data_usage_holt_winters",
    entry_point="gymnos.models.data_usage_holt_winters.DataUsageHoltWinters"
)

models.register(
    name="unusual_data_usage_weighted_thresholds",
    entry_point="gymnos.models.unusual_data_usage_weighted_thresholds.UnusualDataUsageWT"
)

models.register(
    name="keras_classifier",
    entry_point="gymnos.models.keras.KerasClassifier"
)

models.register(
    name="keras_regressor",
    entry_point="gymnos.models.keras.KerasRegressor"
)

models.register(
    name="mte_nn",
    entry_point="gymnos.models.mte_nn.MTENN"
)


# MARK: Preprocessors registration

preprocessors.register(
    name="divide",
    entry_point="gymnos.preprocessors.divide.Divide"
)

preprocessors.register(
    name="grayscale",
    entry_point="gymnos.preprocessors.images.grayscale.Grayscale"
)

preprocessors.register(
    name="image_resize",
    entry_point="gymnos.preprocessors.images.image_resize.ImageResize"
)

preprocessors.register(
    name="replace",
    entry_point="gymnos.preprocessors.replace.Replace"
)

preprocessors.register(
    name="grayscale_to_color",
    entry_point="gymnos.preprocessors.images.grayscale_to_color.GrayscaleToColor"
)

preprocessors.register(
    name="lemmatization",
    entry_point="gymnos.preprocessors.texts.lemmatization.Lemmatization"
)

preprocessors.register(
    name="alphanumeric",
    entry_point="gymnos.preprocessors.texts.alphanumeric.Alphanumeric"
)

preprocessors.register(
    name="tfidf",
    entry_point="gymnos.preprocessors.texts.tfidf.Tfidf"
)

preprocessors.register(
    name="kbest",
    entry_point="gymnos.preprocessors.kbest.KBest"
)

preprocessors.register(
    name="standard_scaler",
    entry_point="gymnos.preprocessors.standard_scaler.StandardScaler"
)

# MARK: Tracker registration

trackers.register(
    name="comet_ml",
    entry_point="gymnos.trackers.comet_ml.CometML"
)

trackers.register(
    name="mlflow",
    entry_point="gymnos.trackers.mlflow.MLflow"
)

trackers.register(
    name="tensorboard",
    entry_point="gymnos.trackers.tensorboard.TensorBoard"
)

# MARK: Data Augmentors registration

data_augmentors.register(
    name="distort",
    entry_point="gymnos.data_augmentors.distort.Distort"
)

data_augmentors.register(
    name="flip",
    entry_point="gymnos.data_augmentors.flip.Flip"
)

data_augmentors.register(
    name="gaussian_distortion",
    entry_point="gymnos.data_augmentors.gaussian_distortion.GaussianDistortion"
)

data_augmentors.register(
    name="greyscale",
    entry_point="gymnos.data_augmentors.greyscale.Greyscale"
)

data_augmentors.register(
    name="histogram_equalisation",
    entry_point="gymnos.data_augmentors.histogram_equalisation.HistogramEqualisation"
)

data_augmentors.register(
    name="invert",
    entry_point="gymnos.data_augmentors.invert.Invert"
)

data_augmentors.register(
    name="random_brightness",
    entry_point="gymnos.data_augmentors.random_brightness.RandomBrightness"
)

data_augmentors.register(
    name="random_color",
    entry_point="gymnos.data_augmentors.random_color.RandomColor"
)

data_augmentors.register(
    name="random_contrast",
    entry_point="gymnos.data_augmentors.random_contrast.RandomContrast"
)

data_augmentors.register(
    name="random_erasing",
    entry_point="gymnos.data_augmentors.random_erasing.RandomErasing"
)

data_augmentors.register(
    name="rotate",
    entry_point="gymnos.data_augmentors.rotate.Rotate"
)

data_augmentors.register(
    name="rotate_range",
    entry_point="gymnos.data_augmentors.rotate_range.RotateRange"
)

data_augmentors.register(
    name="shear",
    entry_point="gymnos.data_augmentors.shear.Shear"
)

data_augmentors.register(
    name="skew",
    entry_point="gymnos.data_augmentors.skew.Skew"
)

data_augmentors.register(
    name="zoom",
    entry_point="gymnos.data_augmentors.zoom.Zoom"
)

data_augmentors.register(
    name="zoom_ground_truth",
    entry_point="gymnos.data_augmentors.zoom_ground_truth.ZoomGroundTruth"
)

data_augmentors.register(
    name="zoom_random",
    entry_point="gymnos.data_augmentors.zoom_random.ZoomRandom"
)
