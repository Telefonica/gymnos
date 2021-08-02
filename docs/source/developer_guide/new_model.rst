New model
==============================

In this tutorial we will create a new model.

The name of the model will be ``my_model`` and the domain will be ``audio/audio_classification``.

First of all, we will run the command :ref:`gymnos-create`:

.. prompt:: bash

    gymnos-create model my_model audio/audio_classification

This wil create the Python module ``gymnos/audio/audio_classification/my_model`` with the following files:

    - ``__init__.py``: entrypoint for the model. It contains the docstring for the module and the public API.
    - ``__model__.py``: gymnos configuration for model
    - ``trainer.py``: trainer class. Here we will define the trainer
    - ``predictor``: predictor class. Here we will define the predictor.
    - ``hydra_conf``: Hydra configuration definition. Here we will define the parameters for the trainer

Defining pip dependencies
--------------------------

You can add any pip dependency needed for your model by editing the ``pip_dependencies`` variable in ``__model__.py``:

.. code-block:: python
    :caption: __model__.py

    pip_dependencies = [
        "torch",
        "torchaudio",
        "scikit-learn<1.0,
    ]

Defining apt dependencies
---------------------------

You can add any ``apt-get`` package needed for your model by editing the ``apt_dependencies`` variable in ``__model__.py``.

.. code-block:: python
    :caption: __model__.py

    apt_dependencies = [
        "ffmpeg",
        "libsm6"
    ]

Defining the public API
------------------------

The ``__init__.py`` will define the public API for your module.

If you open the file you will see something like this:

.. code-block:: python

    from ....utils import lazy_import

    MyModelPredictor = lazy_import("gymnos.audio.audio_classification.my_model.predictor.MyModelPredictor")

By lazy importing the predictor we prevent import errors for dependencies not installed until MyModelPredictor is used.

This public API will allow us to import the predictor like this:

.. code-block:: python

    from gymnos.audio.audio_classification.my_model import MyModelPredictor



Adding the docstring
--------------------

First of all, we will modify the docstring for the module:

.. code-block:: python
    :caption: __init__.py

    """
    Small description about the model
    """

Defining the trainer parameters
--------------------------------

Now we will define the trainer parameters using a `dataclass <https://docs.python.org/3/library/dataclasses.html>`_.

We will add two parameters as example:

    - A required boolean parameter named ``param_1``
    - An optional list parameter named ``param_2`` with default value ``None``

.. code-block:: python
    :caption: hydra_conf.py

    from typing import List
    from dataclasses import dataclass, field


    @dataclass
    class MyModelHydraConf:

        param_1: bool
        param_2: List[str] = None

        _target_: str = field(init=False, default="gymnos.audio.audio_classification.trainer.MyModelTrainer")

The ``_target_`` parameter is mandatory and must default to the path of the trainer. This will be automatically defined by ``gymnos-create``. It is used by `Hydra <https://hydra.cc/docs/next/advanced/instantiate_objects/overview/>`_.

Implementing the trainer
------------------------

First of all, we will write a class docstring explaining about the data structure expected by model and the class parameters.

.. code-block:: python
    :caption: trainer.py

    @dataclass
    class MyModelTrainer(MyModelHydraConf, BaseTrainer):
        """
        Trainer expects a directory for each class where each directory contains the audio samples in .wav format.

        .. code-block::

            class1/
                audio1.wav
                audio2.wav
                ...
            class2/
                audio1.wav
                audio2.wav
                ....

        Parameters
        -------------
        param_1:
            Description about param_1
        param_2:
            Description about param_2
        """

Once the docstring has been written, we will implement the following methods:

    - ``setup(root)``: optional, method called with data directory as parameter
    - ``train()``: required, execute training for model
    - ``test()``: optional, execute testing for model. This method will be called after training.

Constructor
*************

Any parameter defined on ``MyModelHydraConf`` will be available using ``self``, e.g ``self.param_1``.
For any other variable you want to initialize, you can use the method ``__post_init`` from the `dataclass <https://docs.python.org/3/library/dataclasses.html>`_.

.. code-block:: python

    @dataclass
    class MyModelTrainer(MyModelHydraConf, BaseTrainer):

        def __post_init__(self):
            self._param_3 = self.param_2 + ["hello"]

Setup
*******

This method will be called with the directory where data is stored.

.. code-block:: python

    from glob import glob

    @dataclass
    class MyModelTrainer(MyModelHydraConf, BaseTrainer):

        def setup(root):
            self._audio_fpaths = glob(os.path.join(root, "*", "*.wav"))  # we will save all audio file paths


Train
*********

This method must implement the training code and the checkpoint saving

.. code-block:: python

    @dataclass
    class MyModelTrainer(MyModelHydraConf, BaseTrainer):

        def train():
            ...   # Execute training
            mlflow.log_artifact(...)

Any artifact saved using `mlflow.log_artifact <https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_artifact>`_
will be available to the predictor so make sure you save your model weights using ``mlflow.log_artifact``.
Refer to `mlflow documentation <https://www.mlflow.org/docs/latest/python_api/mlflow.html>`_ for more information.

Test
*******

This method must implement the testing code.

.. code-block:: python

    @dataclass
    class MyModelTrainer(MyModelHydraConf, BaseTrainer):

        def test():
            ...   # Execute testing


Implementing the predictor
----------------------------

The predictor is the class responsible for end-to-end predictions.
You must implement two methods:

    - ``load(self, artifacts_dir)``: load weights from artifacts directory. This directory will contain any artifact saved by the trainer using ``mlflow.log_artifact``.
    - ``predict(self, *args, **kwargs)``: method responsible for predictions. Parameters are not defined so it's up to you.

Constructor
************

Optionally, you can add any parameter to the constructor:

.. code-block:: python

    class MyModelPredictor(BasePredictor):

        def __init__(self, param_1, param_2):
            ...

It will be instantiated as follows:

.. code-block:: python

    predictor = MyModelPredictor.from_pretrained(<MLFLOW_RUN_ID_OR_SOFIA_MODEL>, param_1=..., param_2=...)

Load weights
***************

This method must implement the weight loading from the directory containing all artifacts saved by the trainer.

.. code-block:: python

    class MyModelPredictor(BasePredictor):

        def load(self, artifacts_dir):
            # Load model from artifacts directory


.. tip::
    You can access the original config from the trainer with the ``info`` property:

    .. code-block:: python

        def log(self, artifacts_dir):
            self.info.trainer.config

Predict
***********

This method must implement the end-to-end predictions. Parameters for this method will be defined by you.

.. note::
    Parameters should be framework-agnostic, e.g instead of having a ``torch.tensor`` as a parameter,
    you can use a ``np.array`` and then convert it to ``torch.tensor``.
    As a general rule, a Python primitive or a NumPy array is a safe bet but you can also include multiple options
    e.g a ``np.ndarray`` or a ``torch.tensor`` -> ``Union[np.ndarray, torch.tensor]``.

.. code-block:: python

    import numpy
    from typing import Union


    class MyModelPredictor(BasePredictor):

        def predict(self, audio: Union[str, np.ndarray]):
            """
            Predict class from audio

            Parameters
            -----------
            audio:
                Filepath or numpy array
            """
            if isinstance(audio, str):
                # load audio from filepath
            ...

Running the model
--------------------

Once finished, you can run your model with Hydra using the command ``gymnos-train``:

.. prompt:: bash

    gymnos-train trainer=audio.audio_classification.my_model trainer.param_1=false trainer.param_2="[dog,cat]"

.. tip::
    You can use ``dataset=dummy`` to check that your model is working properly, e.g:

    .. prompt:: bash

        gymnos-train trainer=audio.audio_classification.my_model dataset=dummy



Documentation
---------------

Remember to check the :ref:`documentation` for your new model
