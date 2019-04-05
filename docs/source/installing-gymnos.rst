.. index:: ! installing

.. _installing-gymnos:

################################
Installing Gymnos
################################

Installing Gymnos is pretty simple. Here is a step by step plan on how to do it.

First, obtain `Python <https://www.python.org/downloads/>`_ and 
`Pipenv <https://github.com/pypa/pipenv>`_ if you do not already have them. Using Pipenv will make the installation and execution 
easier as it creates and manages a virtualenv for your projects, as well as install required dependencies. You will also need `Git <https://git-scm.com/downloads>`_ in order to clone the repository.

Once you have these, clone the repository:

.. code-block:: bash

   git clone https://github.com/Telefonica/gymnos.git
   cd gymnos/src

.. note::
   If you want to help developing Gymnos, start working at ``devel`` branch

Install required dependencies:

.. code-block:: bash

  pipenv install

Install ``tensorflow-gpu`` to run in GPU:

.. code-block:: bash

  pipenv install tensorflow-gpu

Or ``tensorflow`` to run in CPU:

.. code-block:: bash

  pipenv install tensorflow

You're now ready to run gymnos. Gymnos ships with some example experiments that should get you up and running quickly.

To actually get gymnos running, do the following:

.. code-block:: bash

  pipenv run python3 -m bin.scripts.gymnosd -c experiments/boston_housing.json

This will run an experiment for Boston Housting dataset.

Docker
==========

We provide up to date docker builds for different execution environments and working modalities.

.. note::
  Please make sure `Docker <https://docs.docker.com/v17.12/install/>`_  is install in your computer

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

  docker run gymnos -c <gymnos_training_configuration>


GPU version.

.. code-block:: bash

  nvidia-docker run gymnos-gpu -c <gymnos_training_configuration>

.. note::

    If you want to add new features or try new experiments, the docker environment is the perfect place to do it.
    Simply access the container and you will have all the dependencies resolved to execute your new project with:

    .. code-block:: bash

        python3 -m bin.scripts.gymnosd -c <training_configuration>

    To access your container, run the following command:

    .. code-block:: bash

        docker run -it --entrypoint=/bin/bash gymnos

    Or if you have a GPU:

    .. code-block:: bash

        nvidia-docker run -it --entrypoint=/bin/bash gymnos-gpu
