# Gymnos
[![Build Status](https://dev.azure.com/pablolopezcoya/gymnos/_apis/build/status/Telefonica.gymnos?branchName=master)](https://dev.azure.com/pablolopezcoya/gymnos/_build/latest?definitionId=3&branchName=master)
 [![Python Version](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-369/)
 [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Gymnos is a model training library in Python for Machine Learning. It aims to define conventions in the APIs of the basic components of any supervised learning system such as datasets, models, preprocessors or trackers and therefore be able to execute the training of any model in a simple and automatic way.

## Installation

Gymnos is written in Python.

Install using `pip` from the command line:
```sh
$ pip install git+https://github.com/Telefonica/gymnos.git
```

See the documentation for building from source.

## Quick Start Guide and Usage
To begin, instantiate a `Trainer` object and define the components of your supervised learning system:

```py
import gymnos

# Instantiate a Gymnos trainer definining ML system configuration

dataset = gymnos.core.Dataset(
    dataset=dict(
        type="dogs_vs_cats"
    ),
    one_hot=True,
    preprocessors=[
        dict(
            type="image_resize",
            width=80,
            height=80
        ),
        dict(
            type="grayscale"
        ),
        dict(
            type="divide",
            factor=255.0
        )
    ],
    data_augmentors=[
        dict(
            type="zoom",
            probability=0.3,
            min_factor=0.1,
            max_factor=0.5
        )
    ],
    samples=dict(
        train=0.8,
        test=0.2
    )
)

model = gymnos.core.Model(
    model=dict(
        type="dogs_vs_cats_cnn",
        input_shape=[80, 80, 1],
        classes=2
    ),
    training=dict(
        epochs=10,
        batch_size=32,
        validation_split=0.25
    )
)

tracking = gymnos.core.Tracking(
    trackers=[
        dict(
            type="mlflow",
        ),
        dict(
            type="tensorboard"
        )
    ]
)

trainer = gymnos.trainer.Trainer(model, dataset, tracking)

# Start training
trainer.train()

# Save trainer
trainer.save("saved_trainer.zip")
```

Now you can easily make predictions with your trained ML system:
```py
import gymnos

# Load saved trainer
trainer = gymnos.trainer.Trainer.load("saved_trainer.zip")

# Load dataset
dogs_vs_cats = gymnos.datasets.load("dogs_vs_cats")

# Download dataset files to downloads directory
dl_manager = gymnos.services.DownloadManager("downloads")
dogs_vs_cats.download_and_prepare(dl_manager)

# Load samples into memory
X, y = dogs_vs_cats.load()

# Predict values and probabilities
predictions = trainer.predict(X)
probabilities = trainer.predict_proba(X)
```

### Command Line

You can train your ML system directly from your command line defining your supervised learning system in a JSON file like you would do in code:

```sh
$ gymnos train experiment.json
```

And then get predictions using the saved trainer:
```sh
$ gymnos predict saved_trainer.zip --image cat.png
```

Or run a server to compute predictions:
```sh
$ gymnos serve saved_trainer.zip
```

## Examples and documentation

Gymnos comes with a range of example [Jupyter notebooks](examples/notebooks) and [configurations](examples/experiments) for different experiments.

For instance, to run a configuration for [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats), execute the following line:
```sh
$ gymnos train examples/experiments/dogs_vs_cats.json
```

For more information check out [Gymnos documentation](http://dev-aura-comp-01/docs/gymnos/en/latest/).

## Development
Gymnos is still under development. Contributions are always welcome!.

## License
Gymnos is licensed under [GNU General Public License v3.0](LICENSE)

## Tests
To run the automated tests, clone the repository and run:
```sh
$ pytest
```
from the command line.

## Maintainers
[@kawaits](https://github.com/kawaits)
