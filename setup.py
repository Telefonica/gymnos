import os
import ast
import glob
import fnmatch
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
        if assign.targets and isinstance(assign.value, ast.List) and assign.targets[0].id == "requirements":
            dependencies = []
            for elem in assign.value.elts:
                if isinstance(elem, ast.Str):
                    dependencies.append(elem.s)

    return dependencies


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def remove_suffix(text, suffix):
    if suffix and text.endswith(suffix):
        return text[:-len(suffix)]
    return text


def find_model_dependencies():
    app_dir = os.path.join(here, "gymnos")

    dependencies_by_model = {}

    for path in glob.iglob(os.path.join(app_dir, "**", "__model__.py"), recursive=True):
        with open(path, "r") as fp:
            code = fp.read()

        tree = ast.parse(code)

        assigns = [x for x in tree.body if isinstance(x, ast.Assign)]

        dependencies = None

        for assign in assigns:
            if not assign.targets:
                continue
            if isinstance(assign.value, ast.List) and assign.targets[0].id == "requirements":
                dependencies = []
                for elem in assign.value.elts:
                    if isinstance(elem, ast.Str):
                        dependencies.append(elem.s)

        name = os.path.relpath(path, app_dir)
        name = os.path.dirname(name)
        name = name.replace(os.path.sep, ".")

        dependencies_by_model[name] = dependencies

    return dependencies_by_model


INSTALL_REQUIRES = [
    "rich",
    "mlflow",
    "click",
    "fastdl",
    "dacite",
    "requests",
    "GitPython",
    "omegaconf",
    "stringcase",
    "lazy-object-proxy",
    "hydra-core>=1.1.0",
]

EXTRAS_REQUIRE = {
    "style": [
        "flake8"
    ],
    "docs": [
        "Sphinx",
        "sphinx-tabs",
        "sphinx-click",
        "sphinx-prompt",
        "python-benedict",
        "sphinx-rtd-theme",
        "sphinx-autobuild",
        "sphinxcontrib.asciinema",
        "Sphinx-Substitution-Extensions",
        "sphinx-multiversion"
    ],
}

EXTRAS_REQUIRE.update(find_model_dependencies())

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
    python_requires=">=3.7",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "gymnos-train=gymnos.cli.train:hydra_entry",
            "gymnos-upload=gymnos.cli.upload:main",
            "gymnos-login=gymnos.cli.login:main",
            "gymnos-create=gymnos.cli.create:main"
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
