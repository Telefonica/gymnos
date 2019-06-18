import setuptools

with open("README.md", "r") as fp:
    long_description = fp.read()

DATASET_FILES = [
    "datasets/tiny_imagenet_labels.txt"
]

setuptools.setup(
    name="gymnos",
    version="0.1dev",
    author="Telefonica",
    description="A training platform for AI models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Telefonica/gymnos",
    packages=setuptools.find_packages(where="gymnos"),
    package_data={
        "gymnos": DATASET_FILES
    },
    python_requires=">=3.5",
    install_requires=[
        "numpy",
        "pillow",
        "keras",
        "tqdm",
        "kaggle",
        "numpy",
        "pandas",
        "scikit-learn",
        "spacy",
        "comet-ml",
        "mlflow",
        "statsmodels",
        "py-cpuinfo",
        "gputil",
        "dill",
        "scipy!=1.3",
        "commentjson",
        "pillow"
    ],
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
)
