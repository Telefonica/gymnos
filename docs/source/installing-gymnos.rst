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
    docker-compose build

If you are lucky enough to have a GPU, you just need to execute the following command
to use the GPU in your Docker image.  

.. code-block:: bash
    docker-compose -f docker-compose.yml -f docker-compose.gpu.yml build

.. warning::

   Make sure you meet the following dependencies:

   * docker version:      18.09.1 (or higher)
   * CUDA version:        10.0
   * GPU docker support:  CUDA version compatible

Pull image
-----------

If you just want to get the latest docker build:

.. code-block:: bash

    docker pull telefonica/gymnos:devel-2019-02-12-15-25-29
    docker pull telefonica/gymnos:devel-gpu-2019-02-12-15-25-29

Run image
-------------------

.. code-block:: bash
    docker-compose run gymnos -c <training_configuration>

GPU version.

.. code-block:: bash
    docker-compose -f docker-compose.yml -f docker-compose.gpu.yml run gymnos -c <training_configuration>

.. note::

   Previous example was executed in a GPU environment with the following settings:

   * NVIDIA-SMI:          410.79
   * Driver Version:      410.79
   * CUDA Version:        10.0


Clone the Repository
--------------------

To clone the source code, execute the following command:

.. code-block:: bash

    git clone --recursive https://github.com/Telefonica/gymnos.git
    cd gymnos

If you want to help developing Gymnos, start working at ``devel`` branch
