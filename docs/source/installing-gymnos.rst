.. index:: ! installing

.. _installing-gymnos:

################################
Installing Gymnos
################################

We offer two different ways to install Gymnos (Python and Docker).

Python
==========

First, obtain `Python 3 <https://www.python.org/downloads/>`_ if you do not already have it. It is recommended to setup a virtual environment with `virtualenv <https://github.com/pypa/virtualenv>`_ or `Pipenv <https://github.com/pypa/pipenv>`_ .

First, clone repository:

.. code-block:: bash

  git clone https://github.com/Telefonica/gymnos/tree/devel

In order to make Gymnos as light as possible and at the same time have the maximum possible flexibility when developing Gymnos, we will only install core dependencies:

.. code-block:: bash

  pip3 install -e .

but we also provide 2 possible additional installations:

**a) Install all dependencies**

.. code-block:: bash

  pip3 install -e .[complete]

**b) Install specific module dependency**

.. code-block:: bash

  pip3 install -e .[<type>.<module>]

For example, to install dependencies for tracker ``"mflow"``: :class:`lib.trackers.mlflow.MLFlow`.

.. code-block:: bash

  pip3 install -e .[trackers.mlflow]

Check ``setup.py`` file to see module dependencies.


Note that TensorFlow must be installed manually. Either:

.. code-block:: bash

  pip3 install .[tensorflow]

Or

.. code-block:: bash

  pip3 install .[tensorflow-gpu]

depending on whether you have a GPU or CPU only.

To get started, let's run Gymnos CLI to check if installation was successful. We will solve Boston Housing dataset `Boston Housing dataset <https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html>`_

.. code-block:: bash

  gymnos train experiments/examples/boston_housing.json


Docker
==========

We provide up to date docker builds for different execution environments and working modalities.

.. note::
  Please make sure `Docker <https://docs.docker.com/v17.12/install/>`_  is install in your computer.
  Refer to the following links for instructions about downloading and installing Docker on different platforms:

    - `Docker on Windows 10 <https://runnable.com/docker/install-docker-on-windows-10>`_
    - `Docker on Linux <https://runnable.com/docker/install-docker-on-linux>`_
    - `Docker on macOS <https://runnable.com/docker/install-docker-on-macos>`_

Build image
-----------

If you are a developer and want to build the gymnos image from scratch, choose a Dockerfile that suits 
your development environment.

First, clone repository:

.. code-block:: bash

  git clone https://github.com/Telefonica/gymnos/tree/devel

Then build Docker container:

.. code-block:: bash

  docker build -t gymnos .

If you are lucky enough to have a GPU, you just need to execute the following command to use the GPU in your Docker image.  

.. code-block:: bash

  docker build -f Dockerfile.gpu -t gymnos-gpu .

.. warning::

   Make sure you meet the following dependencies:

   * docker version:      18.09.1 (or higher)
   * CUDA version:        10.0
   * GPU docker support:  CUDA version compatible

.. note::
   Previous example was executed in a GPU environment with the following settings:

   * NVIDIA-SMI:          410.79
   * Driver Version:      410.79
   * CUDA Version:        10.0


Pull image from Telefonica Artifactory
-----------------------------------------

If you just want to get the latest docker build:

First you need to log in to Telefonica Artifactory at dockerhub.hi.inet

.. code-block:: bash

  docker login dockerhub.hi.inet

.. note::

  Please provide your corporate credentials for <USER_ID> and <USER_PASSWORD>

Then pull the latest image:

.. code-block:: bash

  docker pull dockerhub.hi.inet/dcip/aura-prototypes/gymnos

or for gpu environments:

.. code-block:: bash

  docker pull dockerhub.hi.inet/dcip/aura-prototypes/gymnos:gpu-latest

Now check that Docker images were indeed successfully pulled. You should see something like this

.. code-block:: bash

  docker images  

  REPOSITORY                                                 TAG                 IMAGE ID            CREATED             SIZE
  dockerhub.hi.inet/dcip/aura-prototypes/gymnos              gpu-latest          4a55d3c18419        18 minutes ago      4.54GB
  dockerhub.hi.inet/dcip/aura-prototypes/gymnos              latest              37d2d2b9cd0a        18 minutes ago      2.54GB
  tensorflow/tensorflow                                      1.12.0-gpu-py3      413b9533f92a        5 months ago        3.35GB
  tensorflow/tensorflow                                      1.12.0-py3          39bcb324db83        5 months ago        1.33GB


Run image
-------------------

.. code-block:: bash

  docker run -it gymnos


GPU version.

.. note::
  Please make sure `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_  is install in your computer.
  Refer to the following link for instructions about downloading and installing `nvidia-docker on Ubuntu 18.04 <https://cnvrg.io/how-to-setup-docker-and-nvidia-docker-2-0-on-ubuntu-18-04/>`_

.. code-block:: bash

  nvidia-docker run -it gymnos-gpu

To get started let's run Gymnos CLI to check if installation was successful. We will solve Boston Housing dataset `Boston Housing dataset <https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html>`_


Developing for gymnos
=======================

We provide a ``Pipfile`` with all dependencies resolved for the project (including dev dependencies). First obtain `Pipenv <https://github.com/pypa/pipenv>`_ and then install Gymnos with development dependencies:

.. code-block:: bash

  pipenv install --dev

If you have not installed TensorFlow, install it with:

.. code-block:: bash

  pipenv run pip3 install .[tensorflow]

Or ``tensorflow-gpu`` for GPU environments:

.. code-block:: bash

  pipenv run pip3 install .[tensorflow-gpu]

Then, enter virtual environment:

.. code-block:: bash

  pipenv shell

Run tests
----------

.. code-block:: bash

  pytest

To also run slow tests:

.. code-block:: bash

  pytest --runslow

Build documentation
--------------------

.. code-block:: bash

  pipenv run sphinx

The docker environment has all the dependencies resolved to execute your new project with:

.. code-block:: bash

    python3 -m bin.scripts.gymnosd -c <config_path>
