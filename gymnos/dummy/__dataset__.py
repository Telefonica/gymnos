#
#
#   Dummy dataset
#
#

from ..base import BaseDataset

from dataclasses import dataclass, field


@dataclass
class DummyHydraConf:

    _target_: str = field(init=False, default="gymnos.dummy.__dataset__.Dummy")


class Dummy(BaseDataset):

    def __call__(self, root):
        pass


hydra_conf = DummyHydraConf
