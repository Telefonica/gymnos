#
#
#   Tiny Audio Hydra configuration
#
#

from dataclasses import dataclass, field


@dataclass
class TinyAudioHydraConf:

    # Available: door,brokenGlass,dog,cat,baby
    sounds_to_detect: str = "bebe,gato,other,perro,puertas,vidrio"

    window_stride: int = 20
    max_quantization: float = 0.0
    min_quantization: float = 26.0
    sample_rate:  int = 16000
    clip_size: int = 1000
    channels: int = 1
    window_size: float = 30.0
    feature_bin: int = 40
    background_frequency: float = 0.8
    volume_range: float = 1.0
    time_shift: float = 100.0
    validation_percentage: float = 0.1
    test_percentage: float = 10
    train_steps: int = 12000
    learning_rate: float = 0.001
    random_seed: int = 24

    _target_: str = field(init=False, repr=False, default="gymnos.audio.tiny_audio_classification.tiny_audio."
                          "trainer.TinyAudioTrainer")
