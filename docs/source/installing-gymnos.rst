.. index:: ! installing

.. _installing-gymnos:

################################
Installing Gymnos
################################

We offer two different ways to install Gymnos (Python and Docker).

Python
==========

First, obtain `Python 3 <https://www.python.org/downloads/>`_ and `Pipenv <https://github.com/pypa/pipenv>`_ (``pip install pipenv``) if you do not already have them. Using Pipenv will make the installation and execution easier as it creates and manages a virtualenv for your projects, as well as install required dependencies.
To set up an isolated environment and install dependencies just run:

.. code-block:: bash

  pipenv sync

However, note that TensorFlow must be installed manually. Either:

.. code-block:: bash

  pipenv run pip install tensorflow

Or

.. code-block:: bash

  pipenv run pip install tensorflow-gpu

depending on whether you have a GPU. (If you run into problems, try TensorFlow 1.13.1)

Finally, before running any of the scripts, enter the environment with:

.. code-block:: bash

  pipenv shell

The execution directory is located at ``src``:

.. code-block:: bash

  cd src

You're now ready to run gymnos. You can execute an experiment by running:

.. code-block:: bash

  python3 -m bin.scripts.gymnosd -c <config_path>

Gymnos ships with some example experiments that should get you up and running quickly. To actually get gymnos running, do the following:

.. code-block:: bash

  python3 -m bin.scripts.gymnosd -c experiments/examples/boston_housing.json

This will run an experiment for Boston Housting dataset.

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


The docker environment has all the dependencies resolved to execute your new project with:

.. code-block:: bash

    python3 -m bin.scripts.gymnosd -c <config_path>
