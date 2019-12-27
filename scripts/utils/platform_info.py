#
#
#   Platform info
#
#

import subprocess


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
