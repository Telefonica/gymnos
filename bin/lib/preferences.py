#
#
#   Preferences
#
#

import os

from gymnos.utils.io_utils import read_from_json

DEFAULT_PREFERENCES_CONFIG_PATH = os.path.join("bin", "config", "preferences.json")
LOCAL_PREFERENCES_CONFIG_PATH = os.path.join("bin", "config", "preferences.local.json")


def read_preferences():
    """
    Read default preferences and override properties defined by local preferences

    Returns
    -------
    config: dict
        Dictionnary with preferences
    """
    config = read_from_json(DEFAULT_PREFERENCES_CONFIG_PATH, with_comments_support=True)

    if os.path.isfile(LOCAL_PREFERENCES_CONFIG_PATH):
        local_preferences_config = read_from_json(LOCAL_PREFERENCES_CONFIG_PATH, with_comments_support=True)
        config.update(local_preferences_config)

    # Windows and Unix systems use different path separators, we normalize paths to adjust paths
    # separators to current system
    keys_to_normalize = ["download_dir", "extract_dir", "hdf5_datasets_dir", "executions_dir", "trackings_dir",
                         "trained_model_dir", "trained_preprocessors_filename", "training_config_filename",
                         "execution_results_filename"]

    for key in keys_to_normalize:
        if config[key] is not None:
            config[key] = os.path.expanduser(os.path.normpath(config[key]))

    return config
