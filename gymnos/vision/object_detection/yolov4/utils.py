#
#
#   Utils
#
#

import os
import fastdl

from ....config import get_gymnos_home


def download_pretrained_model() -> str:
    return fastdl.download(
        url="http://obiwan.hi.inet/public/gymnos/yolov4/yolov4.conv.137.pth",
        dir_prefix=os.path.join(get_gymnos_home(), "downloads", "yolov4")
    )
