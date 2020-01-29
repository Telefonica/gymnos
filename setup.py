import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r") as fp:
    long_description = fp.read()

about = {}
with open(os.path.join(here, "gymnos", "__version__.py"), "r") as f:
    exec(f.read(), about)

DATASET_FILES = [
    "datasets/tiny_imagenet_labels.txt",
    "datasets/reddit_self_post_classification_labels.txt"
]

VAR_FILES = [
    "models/var/keras_modules.json"
]

REQUIRED_DEPENDENCIES = [
    "numpy",
    "pandas",
    "dill",
    "tqdm",
    "gputil",
    "py-cpuinfo",
    "h5py"
]

EXTRAS_REQUIRE = {
    "serve": [
        "flask"
    ],
    "tests": [
        "pytest",
        "pytest-mock",
        "requests-mock"
    ],
    "docs": [
        "sphinx",
        "sphinx-rtd-theme",
        "sphinx-argparse",
        "sphinx-autobuild"
    ]
}

EXTRAS_REQUIRE["complete"] = sorted(set(sum(EXTRAS_REQUIRE.values(), [])))

SETUP_REQUIRES = [
    "flake8"
]

setuptools.setup(
    name="gymnos",
    version=about["__version__"],
    author=about["__author__"],
    description=about["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=about["__url__"],
    license=about["__license__"],
    packages=setuptools.find_packages(),
    package_data={
        "gymnos": DATASET_FILES + VAR_FILES,
        "scripts": ["config/logging.json"]
    },
    python_requires=">=3.6",
    install_requires=REQUIRED_DEPENDENCIES,
    extras_require=EXTRAS_REQUIRE,
    setup_requires=SETUP_REQUIRES,
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
