#!/usr/bin/python
# -*- coding: utf8 -*-

import os, sys, json, traceback
from trainer import Trainer

import logging
import logging.config
import argparse

BASE_PATH = '/home/sysadmin/gymnos/'
CD_LOG_CONFIG_PATH = BASE_PATH + 'config/logging.json'
SYS_CONFIG_PATH = BASE_PATH + 'config/system.json'
FOLDER_PATH = BASE_PATH + 'data/json'

with open(SYS_CONFIG_PATH, 'rb') as fp:
  sys_config = json.load(fp)

def setup_logging(
    default_path=CD_LOG_CONFIG_PATH,
    default_level=logging.INFO,
    env_key='LOG_CFG'
):
    """Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

    logger = logging.getLogger('gymnosd')
    logger.propagate = False  

    return logger


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", "--training_config", help="sets training configuration file path", action='store')
  config = parser.parse_args()
  
  log = setup_logging()
  TRAINING_CONFIG_PATH = BASE_PATH + config.training_config
  with open(TRAINING_CONFIG_PATH, 'rb') as fp:
    training_config = json.load(fp)
  tr = Trainer(training_config)
  try:
    log.info('---------------------- GYMNOS ENVIRONMENT STARTED ---------------------}')
    tr.run()
    
  except Exception as e:
    log.error('Exception {0}'.format(e))
    traceback.print_exc()

  finally:
    del(tr)
