#
#
#   Neural Network Classifier Hydra configuration
#
#

from dataclasses import dataclass, field


@dataclass
class NeuralNetworkClassifierHydraConf:

    # TODO: define trainer parameters
    batch_size: int = 16
    epochs: int = 100
    min_delta: float = 0.01
    patience: int = 30
    verbose: int = 1
    activation_input: str = 'relu'
    dropout1_rate: float = 0.05
    dropout2_rate: float = 0.1
    activation_hidden1: str = 'relu'
    activation_output: str = 'softmax'
    _target_: str = field(init=False, repr=False, default="gymnos.tabular.classification.neural_network_classifier."
                                                          "trainer.NeuralNetworkClassifierTrainer")
