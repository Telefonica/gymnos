import platform

import GPUtil
import cpuinfo


def platform_details(key):
    cpu_info = cpuinfo.get_cpu_info()

    gpus_info = []
    for gpu in GPUtil.getGPUs():
        gpus_info.append({
            "name": gpu.name,
            "memory": gpu.memoryTotal
        })

    platform_details = {
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "platform": platform.platform(),
        "system": platform.system(),
        "node": platform.node(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "cpu": {
            "brand": cpu_info["brand"],
            "cores": cpu_info["count"]
        },
        "gpu": gpus_info
    }

    return platform_details[key]