#
#
#   Lenet Audio Hydra configuration
#
#

from dataclasses import dataclass, field


@dataclass
class LenetAudioHydraConf:

    optimizer: str = 'adam'
    momentum: float = 0.9
    weight_decay: float = 0.0
    seed: int = 2021
    epochs: int = 80
    batch_size: int = 12
    patience: int = 10
    lr: float = 0.001

    window_size: float = 3.0
    sampling_rate: int = 16000

    cuda: bool = True
    balance: bool = True

    num_classes: int = 5

    _target_: str = field(init=False, repr=False, default="gymnos.audio.audio_classification.lenet_audio."
                                                          "trainer.LenetAudioTrainer")
