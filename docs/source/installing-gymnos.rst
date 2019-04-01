.. index:: ! installing

.. _installing-gymnos:

################################
Installing Gymnos
################################

Docker
==========

We provide up to date docker builds for different execution environments and working modalities.

Build image
-----------

If you are a developer and want to build the gymnos image from scratch, choose a Dockerfile that suits 
your development environment.

.. code-block:: bash
    docker build -t gymnos .

If you are lucky enough to have a GPU, you just need to execute the following command
to use the GPU in your Docker image.  

.. code-block:: bash
    docker build -f Dockerfile.gpu -t gymnos-gpu .

.. warning::

   Make sure you meet the following dependencies:

   * docker version:      18.09.1 (or higher)
   * CUDA version:        10.0
   * GPU docker support:  CUDA version compatible

   Previous example was executed in a GPU environment with the following settings:

   * NVIDIA-SMI:          410.79
   * Driver Version:      410.79
   * CUDA Version:        10.0


Pull image
-----------

If you just want to get the latest docker build:

.. code-block:: bash



Run image
-------------------

.. code-block:: bash
    docker run gymnos -c <training_configuration>

GPU version.

.. code-block:: bash
    nvidia-docker run gymnos-gpu -c <training_configuration>

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


Clone the Repository
--------------------

To clone the source code, execute the following command:

.. code-block:: bash

    git clone --recursive https://github.com/Telefonica/gymnos.git
    cd gymnos

If you want to help developing Gymnos, start working at ``devel`` branch
