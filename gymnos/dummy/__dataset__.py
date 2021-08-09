#
#
#   Dummy dataset
#
#

from ..base import BaseDataset

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DummyDatasetHydraConf:

    path: Optional[str] = None

    _target_: str = field(init=False, default="gymnos.dummy.__dataset__.DummyDataset")


class DummyDataset(DummyDatasetHydraConf, BaseDataset):

    def download(self, root):
        pass


hydra_conf = DummyDatasetHydraConf

pip_dependencies = []

apt_dependencies = []
