#
#
#   Hello World Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class HelloWorldHydraConf:

    # TODO: add custom parameters

    _target_: str = field(init=False, repr=False, default="gymnos.datasets.hello_world.dataset.HelloWorld")
