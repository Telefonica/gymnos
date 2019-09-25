#
#
#   Test trainer
#
#

import os
import pytest

from gymnos.trainer import Trainer
from gymnos.services.download_manager import DownloadManager


BOSTON_HOUSING_SPEC = {
    "model": {
        "model": {
            "type": "keras_regressor",
            "sequential": [
                {"type": "dense", "units": 1, "activation": "linear"}
            ],
            "input_shape": [13],
            "optimizer": "sgd",
            "loss": "mse",
            "metrics": ["mae"]
        },
        "training": {
            "batch_size": 32,
            "epochs": 1,
            "callbacks": [
                {
                    "type": "early_stopping"
                }
            ],
            "validation_split": 0.2
        }
    },
    "dataset": {
        "dataset": {
            "type": "boston_housing"
        },
        "preprocessors": [
            {
                "type": "divide",
                "factor": 100
            }
        ]
    },
    "tracking": {
        "trackers": [
            {
                "type": "tensorboard"
            },
            {
                "type": "mlflow"
            }
        ]
    }
}

DATA_USAGE_HOLT_WINTERS_SPEC = {
    "dataset": {
        "dataset": {
            "type": "data_usage_test"
        },
        "samples": {
            "train": 20,
            "test": 20
        }
    },
    "model": {
        "model": {
            "type": "data_usage_holt_winters"
        }
    }
}

DATA_USAGE_LINEAR_REGRESSION_SPEC = {
    "dataset": {
        "dataset": {
            "type": "data_usage_test"
        },
        "samples": {
            "train": 5,
            "test": 5
        }
    },
    "model": {
        "model": {
            "type": "data_usage_linear_regression"
        }
    }
}

DOGS_VS_CATS_SPEC = {
    "model": {
        "model": {
            "type": "dogs_vs_cats_cnn",
            "input_shape": [32, 32, 3]
        },
        "training": {
            "batch_size": 32,
            "epochs": 1,
            "validation_split": 0.25
        }
    },
    "dataset": {
        "dataset": {
            "type": "dogs_vs_cats"
        },
        "one_hot": True,
        "samples": {
            "train": 5,
            "test": 5
        },
        "preprocessors": [
            {
                "type": "image_resize",
                "width": 32,
                "height": 32
            },
            {
                "type": "divide",
                "factor": 255
            }
        ]
    }
}

IMDB_SPEC = {
    "dataset": {
        "dataset": {
            "type": "imdb"
        },
        "one_hot": True,
        "samples": {
            "train": 5,
            "test": 5
        },
        "preprocessors": [
            {
                "type": "alphanumeric"
            },
            {
                "type": "lemmatization"
            },
            {
                "type": "tfidf",
                "min_df": 0.5
            },
            {
                "type": "kbest",
                "k": 6,
                "scorer": "chi2"
            }
        ]
    },
    "model": {
        "model": {
            "type": "keras_classifier",
            "sequential": [
                {"type": "dense", "units": 2}
            ],
            "input_shape": [6],
            "loss": "categorical_crossentropy",
            "optimizer": "adam",
            "metrics": ["acc"]
        },
        "training": {
            "epochs": 1
        }
    }
}

ROCK_PAPER_SCISSORS_SPEC = {
    "model": {
        "model": {
            "type": "dogs_vs_cats_cnn",
            "input_shape": [80, 80, 1],
            "classes": 3
        }
    },
    "dataset": {
        "dataset": {
            "type": "rock_paper_scissors"
        },
        "one_hot": True,
        "samples": {
            "train": 10,
            "test": 10
        },
        "preprocessors": [
            {
                "type": "image_resize",
                "width": 80,
                "height": 80
            },
            {
                "type": "divide",
                "factor": 255
            }
        ],
        "data_augmentors": [
            {
                "type": "invert",
                "probability": 0.2
            },
            {
                "type": "distort",
                "probability": 0.3,
                "grid_width": 10,
                "grid_height": 10,
                "magnitude": 3
            }
        ]
    },
    "tracking": {
        "trackers": [
            {
                "type": "tensorboard"
            }
        ]
    }
}

SYNTHETIC_DIGITS = {
    "model": {
        "model": {
            "type": "dogs_vs_cats_cnn",
            "input_shape": [50, 50, 3],
            "classes": 10
        },
        "training": {
            "epochs": 2,
            "validation_split": 0.25
        }
    },
    "dataset": {
        "dataset": {
            "type": "synthetic_digits"
        },
        "one_hot": True,
        "preprocessors": [
            {
                "type": "image_resize",
                "width": 50,
                "height": 50
            },
            {
                "type": "divide",
                "factor": 255
            }
        ],
        "samples": {
            "train": 10,
            "test": 5
        }
    }
}


TINY_IMAGENET_SPEC = {
    "dataset": {
        "dataset": {
            "type": "tiny_imagenet"
        },
        "one_hot": True,
        "samples": {
            "train": 5,
            "test": 5
        },
        "chunk_size": 2,
        "data_augmentors": [
            {
                "type": "zoom",
                "probability": 0.3,
                "min_factor": 0.5,
                "max_factor": 1.5
            },
            {
                "type": "flip",
                "probability": 0.4,
                "top_bottom_left_right": "LEFT_RIGHT"
            }
        ],
        "preprocessors": [
            {
                "type": "grayscale"
            },
            {
                "type": "image_resize",
                "width": 25,
                "height": 25
            }
        ]
    },
    "model": {
        "model": {
            "type": "keras_classifier",
            "sequential": [
                {"type": "flatten"},
                {"type": "dense", "units": 200, "activation": "softmax"}
            ],
            "input_shape": [25, 25, 1],
            "loss": "categorical_crossentropy",
            "optimizer": "sgd",
            "metrics": ["acc"]
        }
    }
}

UNUSUAL_DATA_USAGE = {
    "model": {
        "model": {
            "type": "unusual_data_usage_weighted_thresholds"
        }
    },
    "dataset": {
        "dataset": {
            "type": "unusual_data_usage_test"
        },
        "samples": {
            "train": 5,
            "test": 5
        },
        "shuffle": False
    }
}


@pytest.mark.integration
@pytest.mark.parametrize("spec", [UNUSUAL_DATA_USAGE, TINY_IMAGENET_SPEC, SYNTHETIC_DIGITS, ROCK_PAPER_SCISSORS_SPEC,
                                  IMDB_SPEC, DOGS_VS_CATS_SPEC, BOSTON_HOUSING_SPEC, DATA_USAGE_HOLT_WINTERS_SPEC,
                                  DATA_USAGE_LINEAR_REGRESSION_SPEC])
def test_training(spec, tmp_path, session_tmp_path):
    trainer = Trainer.from_dict(spec)

    dl_manager = DownloadManager(download_dir=str(session_tmp_path))

    trainer.train(dl_manager, trackings_dir=os.path.join(str(tmp_path / "trackings")))
