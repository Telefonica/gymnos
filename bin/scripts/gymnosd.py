# !/usr/bin/python
# -*- coding: utf8 -*-

import argparse
import json
import os
import traceback

from lib.log import logger
from lib.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--training_config", help="sets training configuration file path", action='store',
                        required=True)
    config = parser.parse_args()

    with open(os.path.join(config.training_config), 'r') as fp:
        training_config = json.load(fp)

    tr = Trainer(training_config)

    logger = logger.get_logger()

    try:
        logger.info('---------------------- GYMNOS ENVIRONMENT STARTED ---------------------}')
        tr.run()

    except Exception as e:
        logger.error('Exception {0}'.format(e))
        traceback.print_exc()

    finally:
        del (tr)










