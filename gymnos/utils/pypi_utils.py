#
#
#   Pypi utils
#
#

import pkg_resources

from typing import List
from urllib.parse import urlparse

from .py_utils import remove_prefix


class DependencyMissing(Exception):
    pass


def get_missing_dependencies(dependencies):
    missing_dependencies = []
    for dependency in dependencies:
        if dependency.startswith("git+"):
            import_dependency = parse_egg_from_vcs(dependency)
        else:
            import_dependency = dependency

        try:
            pkg_resources.require(import_dependency)
        except pkg_resources.DistributionNotFound:
            missing_dependencies.append(dependency)
    return missing_dependencies


def parse_egg_from_vcs(dependency):
    parsed = urlparse(remove_prefix(dependency, "git+"))
    if parsed.fragment.startswith("egg"):
        return remove_prefix(parsed.fragment, "egg=")
    else:
        raise ValueError(f"Unknown egg {parsed.fragment}")


def assert_dependencies(dependencies: List[str]):
    """
    Check if dependencies are installed. If some dependency is not installed it raises ``DependencyMissing`` error.

    Parameters
    ----------
    dependencies
        List of dependencies to check
    """
    missing = get_missing_dependencies(dependencies)
    if missing:
        raise DependencyMissing(f"The following dependencies are misssing: {', '.join(missing)}. "
                                f"Please install them (`pip install {' '.join(missing)}`)")
