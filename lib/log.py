import json
import logging
import logging.config as conf
import os
import re


class Log(object):
    def __init__(self, default_level=logging.INFO, get_logger='gymnosd'):
        self._get_logger = get_logger
        self._default_level = default_level
        self._default_path = None
        self._sys_config_path = None
        self._log_file_path = None
        self.__set_paths()
        self.logger = self.__run_setup_logging()

    def setup_prefix(self, params):
        if params.__name__ != params.__name__.upper():
            result = "_".join(re.findall('[A-Z][^A-Z]*', params.__name__)).upper()
        else:
            result = params.__name__
        return result

    def get_logger(self):
        return self.logger

    def __run_setup_logging(self):

        if os.path.exists(self._log_file_path):
            os.remove(self._log_file_path)
        if os.path.exists(self._default_path):
            with open(self._default_path, 'rt') as f:
                config = json.load(f)
            conf.dictConfig(config)
        else:
            logging.basicConfig(level=self._default_level)

        logger = logging.getLogger(self._get_logger)

        logger.propagate = False

        return logger

    def __set_paths(self):
        self._sys_config_path = './config/system.json'
        self._default_path = './config/logging.json'

        with open(self._sys_config_path, 'r') as fp:
            sys_config = json.load(fp)

        logging_path = sys_config['paths']['logs']
        logging_filename = sys_config['filenames']['logs']
        self._log_file_path = os.path.join(logging_path, logging_filename)


logger = Log(default_level=logging.INFO, get_logger='gymnosd')
