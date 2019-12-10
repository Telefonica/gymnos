#
#
#   Functions to handle Gymnos configuration values.
#
#

import os
import json
import inspect
import logging

DEFAULT_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".gymnos")
DEFAULT_CONFIG_FILE_NAME = "gymnos.json"

logger = logging.getLogger(__name__)


def _load_config(path_or_file_like):
    if hasattr(path_or_file_like, "read"):
        config = json.load(path_or_file_like)
    else:
        with open(path_or_file_like) as fp:
            config = json.load(fp)
    return config


class RequiredValuesMissing(ValueError):
    """
    Exception when a required value is missing, i.e the config variable is not in an environment
    variable or in a configuration file.
    """


class Value():
    """
    Configuration value

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


class Config:
    """
    Configuration.

    Parameters
    --------------
    files: list of str
        List of files to look for variables. If None load default config file for Gymnos
    """

    def __init__(self, files=None):
        if files is None:
            gymnos_config_dir = os.getenv("GYMNOS_CONFIG_DIR", DEFAULT_CONFIG_DIR)
            files = [os.path.join(gymnos_config_dir, DEFAULT_CONFIG_FILE_NAME)]

        self.files = files

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

    def _build_required_values_missing_error(self, var_names_with_help, files_to_look=None):
        files_to_look = files_to_look or []

        message = inspect.cleandoc("""
            To use this functionality, you need to provide the following required variables: {}.
            You can export them to the environment, e.g:
                $ export VAR_NAME=xxxxxxxxxx
        """)

        if files_to_look:
            message += "\n" + inspect.cleandoc("""
                You can also choose to place this variable with their value in one of the following file paths: {}.
                For your security, ensure that other users of your computer do not have read access to your credentials.
                On unix-based systems you can do this with the following command:
                    $ chmod 600 <file_path>
            """)
        list_vars_msg = ""
        for var_name, var_help in var_names_with_help:
            list_vars_msg += "\n - {}".format(var_name)
            if var_help:
                list_vars_msg += " ({})".format(var_help)

        message = message.format(list_vars_msg, ", ".join([str(file) for file in files_to_look]))

        return message

    def load(self):
        """
        Load values from configuration files or environment variables.

        Raises
        ---------
        RequiredValueMissing
            If a required value is not found.
        """
        existing_config_files = [path for path in self.files if hasattr(path, "read") or os.path.isfile(path)]

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
            error_message = self._build_required_values_missing_error(var_names_with_help=missing_required_vars,
                                                                      files_to_look=self.files)
            raise RequiredValuesMissing(error_message)
