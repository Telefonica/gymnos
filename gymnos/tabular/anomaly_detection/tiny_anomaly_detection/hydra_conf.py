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
    # TODO: define trainer parameters

    _target_: str = field(init=False, repr=False, default="gymnos.tabular.anomaly_detection.tiny_anomaly_detection."
                                                          "trainer.TinyAnomalyDetectionTrainer")
