# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys
import warnings
import packaging.version

from git import Repo
from sphinx.directives.other import TocTree

sys.path.insert(0, os.path.abspath("_ext"))


def get_stable_release():
    repo = Repo(os.path.join("..", ".."))

    releases = []
    for tag in repo.tags:
        tag_version = packaging.version.parse(tag.name)

        if tag_version.is_prerelease:
            continue

        releases.append(tag_version)

    max_release = max(releases)

    return str(max_release)


# -- Project information -----------------------------------------------------

project = 'gymnos'
copyright = '2021, Telefonica'
author = 'Telefonica'

try:
    stable_release = get_stable_release()
except Exception as e:
    warnings.warn(f"Error reading stable release: {e}")
    stable_release = "???"

rst_prolog = f"""
.. |stable_release| replace:: {stable_release}
"""


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_tabs.tabs",
    "sphinx_rtd_theme",
    "autoyaml",
    "experiment_install",
    "sphinx_click",
    "sphinxcontrib.asciinema",
    "sphinx-prompt",
    "sphinx_substitution_extensions",
    "sphinx_multiversion"
]

add_module_names = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# autodoc_typehints = "none"

html_css_files = [
    "custom.css"
]

html_js_files = [
    "custom.js"
]

autodoc_mock_imports = ["tqdm", "numpy", "pandas", "torch", "torchvision", "pytorch_lightning", "torchmetrics",
                        "efficientnet_pytorch", "PIL", "cv2", "tensorflow", "sklearn", "bounding_box", "pycocotools",
                        "supersuit", "stable_baselines3", "atari_py", "gym_ple"]

sphinx_tabs_disable_tab_closing = True

smv_tag_whitelist = r"^(?!0\.1.*).*"

smv_branch_whitelist = "master"


class MiscTocTree(TocTree):

    def run(self):
        rst = super().run()

        for idx, (_, name) in enumerate(rst[0][0]['entries']):
            if "/misc/" in name:
                rst[0][0]['entries'].append(rst[0][0]['entries'].pop(idx))

        return rst


def setup(app):
    # app.connect("doctree-resolved", reverse_toctree)
    app.add_directive('misctoctree', MiscTocTree)
