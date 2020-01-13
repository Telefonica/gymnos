import setuptools

with open("README.md", "r") as fp:
    long_description = fp.read()


def prefix_dict(dict_to_prefix, prefix):
    return {(prefix + key): val for key, val in dict_to_prefix.items()}


DATASET_FILES = [
    "datasets/tiny_imagenet_labels.txt",
    "datasets/reddit_self_post_classification_labels.txt"
]

VAR_FILES = [
    "models/var/keras_modules.json"
]

REQUIRED_DEPENDENCIES = [
    "kaggle",
    "requests",
    "pysmb",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "dill",
    "tqdm",
    "gputil",
    "py-cpuinfo",
    "h5py"
]

DATASET_EXTRAS_DEPENDENCIES = {
    "data_usage_test": [
        "statsmodels"
    ],
    "unusual_data_usage_test": [
        "statsmodels"
    ]
}

PREPROCESSORS_EXTRAS_DEPENDENCIES = {
    "lemmatization": [
        "spacy"
    ],
    "tfidf": [
        "spacy"
    ]
}

MODELS_EXTRAS_DEPENDENCIES = {

}

TRACKERS_EXTRAS_DEPENDENCIES = {
    "comet_ml": [
        "comet-ml"
    ],
    "mlflow": [
        "mlflow"
    ]
}

EXTRAS_REQUIRE = {}

EXTRAS_REQUIRE.update(prefix_dict(DATASET_EXTRAS_DEPENDENCIES, "datasets."))
EXTRAS_REQUIRE.update(prefix_dict(PREPROCESSORS_EXTRAS_DEPENDENCIES, "preprocessors."))
EXTRAS_REQUIRE.update(prefix_dict(MODELS_EXTRAS_DEPENDENCIES, "models."))
EXTRAS_REQUIRE.update(prefix_dict(TRACKERS_EXTRAS_DEPENDENCIES, "trackers."))

EXTRAS_REQUIRE["datasets"] = sorted(set(sum(DATASET_EXTRAS_DEPENDENCIES.values(), [])))
EXTRAS_REQUIRE["models"] = sorted(set(sum(MODELS_EXTRAS_DEPENDENCIES.values(), [])))
EXTRAS_REQUIRE["preprocessors"] = sorted(set(sum(PREPROCESSORS_EXTRAS_DEPENDENCIES.values(), [])))
EXTRAS_REQUIRE["trackers"] = sorted(set(sum(TRACKERS_EXTRAS_DEPENDENCIES.values(), [])))

EXTRAS_REQUIRE["serve"] = ["flask"]
EXTRAS_REQUIRE["deploy"] = ["requests"]

EXTRAS_REQUIRE["complete"] = sorted(set(sum(EXTRAS_REQUIRE.values(), [])))

EXTRAS_REQUIRE["tensorflow"] = ["tensorflow>1.8.0,<2.0.0"]
EXTRAS_REQUIRE["tensorflow_gpu"] = ["tensorflow-gpu>=1.8.0,<2.0.0"]

setuptools.setup(
    name="gymnos",
    version="0.1dev",
    author="Telefonica",
    description="A training platform for AI models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Telefonica/gymnos",
    packages=setuptools.find_packages(),
    package_data={
        "gymnos": DATASET_FILES + VAR_FILES,
        "scripts": ["config/logging.json"]
    },
    python_requires=">=3.6",
    install_requires=REQUIRED_DEPENDENCIES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={"console_scripts": ["gymnos = scripts.cli:main"]},
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
)
