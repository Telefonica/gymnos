import os
import ast
import glob
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r") as f:
    long_description = f.read()

about = {}
with open(os.path.join(here, "gymnos", "__about__.py"), "r") as f:
    exec(f.read(), about)


def parse_dependencies(filepath):
    with open(filepath, "r") as fp:
        code = fp.read()

    tree = ast.parse(code)

    assigns = [x for x in tree.body if isinstance(x, ast.Assign)]

    dependencies = None

    for assign in assigns:
        if assign.targets and isinstance(assign.value, ast.List) and assign.targets[0].id == "dependencies":
            dependencies = []
            for elem in assign.value.elts:
                if isinstance(elem, ast.Str):
                    dependencies.append(elem.s)

    return dependencies


def parse_models_requirements():
    entrypoints = glob.glob(os.path.join(here, "gymnos", "models", "*", "*", "*", "__init__.py"))

    dependencies_by_model = {}
    for entrypoint in entrypoints:
        model_name = os.path.basename(os.path.dirname(entrypoint))
        dependencies = parse_dependencies(entrypoint)
        if dependencies is not None:
            dependencies_by_model[model_name] = dependencies

    return dependencies_by_model


INSTALL_REQUIRES = [
    "rich",
    "mlflow",
    "fastdl",
    "dacite",
    "requests",
    "GitPython",
    "omegaconf",
    "hydra-core>=1.1.0",
    'dataclasses; python_version < "3.7"'
]

EXTRAS_REQUIRE = {
    "docs": [
            "Sphinx"
        ],
}

EXTRAS_REQUIRE.update({f"model.{key}": value for key, value in parse_models_requirements().items()})

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
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "gymnos-train=gymnos.cli.train:hydra_entry",
            "gymnos-upload=gymnos.cli.upload:main"
            "gymnos-login=gymnos.cli.login:main"
        ]},
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
)
