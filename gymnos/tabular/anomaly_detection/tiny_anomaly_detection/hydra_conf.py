#
#
#   Tiny Anomaly Detection Hydra configuration
#
#

from dataclasses import dataclass, field


@dataclass
class TinyAnomalyDetectionHydraConf:

    optimizer: str = 'adam'
    epochs: int = 100
    batch_size: int = 32
    loss: str = 'mse'
    encoding_dim: int = 3
    dropout: float = 0.1
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    sensor_sample_rate: int = 15
    sample_time: int = 1
    samples_per_file: int = 15
    model_name: str = 'TinyAutoencoder'
    tflite_model_name: str = 'TinyAutoencoderLite'

    _target_: str = field(init=False, repr=False, default="gymnos.tabular.anomaly_detection.tiny_anomaly_detection."
                                                          "trainer.TinyAnomalyDetectionTrainer")
