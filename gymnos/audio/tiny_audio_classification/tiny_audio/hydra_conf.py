#
#
#   Tiny Audio Hydra configuration
#
#

from dataclasses import dataclass, field


@dataclass
class TinyAudioHydraConf:

    sounds_to_detect: str = "puertas,otros,_noise"
    channels: int = 1
    columns: int = 32
    epochs: int = 200

    _target_: str = field(init=False, repr=False, default="gymnos.audio.tiny_audio_classification.tiny_audio."
                          "trainer.TinyAudioTrainer")
