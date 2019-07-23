#
#
#   Test experiment
#
#

from gymnos.core import Experiment


def test_experiment_name():
    experiment = Experiment()

    assert len(experiment.name) == 32  # hexadecimal uuid

    experiment = Experiment(name="hello")

    assert experiment.name == "hello"
