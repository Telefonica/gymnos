#
#
#   Pypi utils
#
#

import pkg_resources

from typing import List


class DependencyMissing(Exception):
    pass


def get_missing_dependencies(dependencies):
    missing_dependencies = []
    for dependency in dependencies:
        try:
            pkg_resources.require(dependency)
        except pkg_resources.DistributionNotFound:
            missing_dependencies.append(dependency)
    return missing_dependencies


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
