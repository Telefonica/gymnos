#
#
#   Service
#
#

import os
import json
import logging
import inspect

logger = logging.getLogger(__name__)


def _load_config(self, path):
    with open(path) as fp:
        config = json.load(fp)
    return config


class RequiredValuesMissing(Exception):
    """
    Exception when a required value is missing, i.e the config variable is not in an environment
    variable or in a configuration file.
    """

    def __init__(self, var_names_with_help, files_to_look=None):
        files_to_look = files_to_look or []

        message = inspect.cleandoc("""
            To use this service, you need to provide the following required variables: {}
            You can place this variable with their value in one of the following file paths: {}.
            For your security, ensure that other users of your computer do not have read access to your credentials.
            On unix-based systems you can do this with the following command:
                $ chmod 600 <file_path>
            You can also choose to export them to the environment, e.g:
                $ export VAR_NAME=xxxxxxxxxx
        """)
        list_vars_msg = ""
        for var_name, var_help in var_names_with_help:
            list_vars_msg += "\n - {}".format(var_name)
            if var_help:
                list_vars_msg += " ({})".format(var_help)
        message = message.format(list_vars_msg, ", ".join(files_to_look))
        super().__init__(message)


class Value():
    """
    Configuration value for service.

    Parameters
    ------------
    default: any, optional
        Default value if not present
    help: str, optional
        About this value
    required: bool, optional
        Whether or not the value is required
    """

    def __init__(self, default=None, required=False, help=""):
        self._default = default
        self.help = help
        self.required = required

    @property
    def default(self):
        if callable(self._default):
            return self._default()
        else:
            return self._default

    @default.setter
    def default(self, value):
        self._default = value


class ServiceConfig:
    """
    Service configuration.

    Parameters
    --------------
    files: list of str
        List of files to look for variables
    """

    def __init__(self, files=None):
        self.files = files or []

    def _get_config_var_names(self):
        """
        Get configuration variable names. It looks for class variables with type Value and not starting with __

        Returns
        ---------
        list
            Configuration variable names.
        """
        def is_config_value(var_name):
            var_value = getattr(self, var_name)
            return not callable(var_value) and isinstance(var_value, Value) and not var_name.startswith("__")
        return list(filter(is_config_value, vars(self.__class__)))

    def load(self):
        """
        Load values from configuration files or environment variables.

        Raises
        ---------
        RequiredValueMissing
            If a required value is not found.
        """
        existing_config_files = [path for path in self.files if os.path.isfile(path)]

        if not existing_config_files:
            config = {}
            logger.info("No config file found. Loading with environment variables.")
        else:
            config = _load_config(existing_config_files[0])

        missing_required_vars = []
        for config_var_name in self._get_config_var_names():
            value = getattr(self, config_var_name)

            if config_var_name in os.environ:
                setattr(self, config_var_name, os.environ[config_var_name])
            elif config_var_name in config:
                setattr(self, config_var_name, config[config_var_name])
            elif not value.required:
                setattr(self, config_var_name, value.default)
            else:
                missing_required_vars.append((config_var_name, value.help))

        if missing_required_vars:
            raise RequiredValuesMissing(var_names_with_help=missing_required_vars, files_to_look=self.files)


class Service:
    """
    Base class for all services.
    """

    class Config(ServiceConfig):
        """
        Base config for all services.

        Parameters
        -----------
        download_dir: str, optional
            Directory to download files
        force_download: bool, optional
            Whether or not force download if file exists
        config_files: list of str
            Files to search for configuration variables.
        """

    def __init__(self, download_dir="downloads", force_download=False, config_files=None):
        self.download_dir = download_dir
        self.force_download = force_download

        self.config = self.Config(files=config_files)
        self.config.load()
