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
your development environment:

.. code-block:: bash

    docker build -t gymnos-devel -f Dockerfile.devel .
    docker build -t gymnos-devel-gpu -f Dockerfile.devel-gpu .

Pull image
-----------

If you just want to get the latest docker build:

.. code-block:: bash

    docker pull telefonica/gymnos:devel-2019-02-12-15-25-29
    docker pull telefonica/gymnos:devel-gpu-2019-02-12-15-25-29

Run image
-------------------

.. code-block:: bash

   docker run gymnos-devel -c <training_configuration>

If you are lucky enough to have a GPU for development, you just need to execute the following command
to get your gymnos docker image running on a GPU.  

.. code-block:: bash

   nvidia-docker run -it gymnos-devel-gpu bash

.. note::

   Previous example was executed in a GPU environment with the following settings:

   * NVIDIA-SMI:          410.79
   * Driver Version:      410.79
   * CUDA Version:        10.0



.. warning::

   Make sure you meet the following dependencies:

   * docker version:      18.09.1 (or higher)
   * CUDA version:        10.0
   * GPU docker support:  CUDA version compatible


Clone the Repository
--------------------

To clone the source code, execute the following command:

.. code-block:: bash

    git clone --recursive https://github.com/Telefonica/gymnos.git
    cd gymnos

If you want to help developing Gymnos, start working at ``devel`` branch
