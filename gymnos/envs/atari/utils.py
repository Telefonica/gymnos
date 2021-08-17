#
#
#   Atari utils
#
#


import os
import fastdl

from atari_py.import_roms import import_roms as _import_roms

from ...config import get_gymnos_home


def import_atari_roms():
    dir_path = os.path.join(get_gymnos_home(), "downloads", "atari")

    with fastdl.Parallel(prefer="threads") as p:

        hc_roms_download = p.download(
            url="http://obiwan.hi.inet/public/gymnos/atari/HC_ROMS.zip",
            dir_prefix=dir_path,
            extract=True
        )

        roms_download = p.download(
            url="http://obiwan.hi.inet/public/gymnos/atari/ROMS.zip",
            dir_prefix=os.path.join(get_gymnos_home(), "downloads", "atari"),
            extract=True
        )

        hc_roms_download.get()
        roms_download.get()

    _import_roms(dir_path)
