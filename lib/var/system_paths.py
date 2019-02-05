import json

BASE_PATH = '/home/sysadmin/gymnos/'
SYS_CONFIG_PATH = BASE_PATH + 'config/system.json'

with open(SYS_CONFIG_PATH, 'rb') as fp:
  sys_config = json.load(fp)

DATASETS_PATH = sys_config['paths']['datasets']
MODELS_PATH = sys_config['paths']['models']