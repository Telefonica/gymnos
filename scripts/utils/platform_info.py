#
#
#   Platform info
#
#

import gymnos
import logging
import platform
import subprocess


logger = logging.getLogger(__name__)


def get_cpu_info():
    import cpuinfo

    cpu_info = cpuinfo.get_cpu_info()
    return dict(brand=cpu_info["brand"], cores=cpu_info["count"])


def get_gpus_info():
    import GPUtil

    def retrieve_gpu_info(gpu):
        return dict(name=gpu.name, memory=gpu.memoryTotal)
    return [retrieve_gpu_info(gpu) for gpu in GPUtil.getGPUs()]


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], universal_newlines=True).strip()


def get_platform_info():
    info = dict(platform=platform.platform())

    try:
        info["cpu"] = get_cpu_info()
    except Exception as e:
        logger.error("Error retrieving CPU information: {}".format(e))

    try:
        info["gpu"] = get_gpus_info()
    except Exception as e:
        logger.error("Error retrieving GPU information: {}".format(e))

    info["gymnos"] = dict(version=gymnos.__version__)

    try:
        info["gymnos"]["git_hash"] = get_git_revision_hash()
    except Exception as e:
        logger.error("Error retrieving git revision hash: {}".format(e))

    return info
