#########################
Installation
#########################

pip
=================

First, obtain `Python 3 <https://www.python.org/downloads/>`_ if you do not already have it. It is recommended to setup a virtual environment with `virtualenv <https://github.com/pypa/virtualenv>`_ or `Pipenv <https://github.com/pypa/pipenv>`_ .

Then, you need to clone the repository:

.. code-block:: console

    $ git clone https://github.com/Telefonica/gymnos/tree/devel
    $ cd gymnos

To install Gymnos:

.. code-block:: console

    $ pip3 install -e .

Note that TensorFlow must be installed manually. Either:

.. code-block:: console

  $ pip3 install .[tensorflow]

Or

.. code-block:: console

  $ pip3 install .[tensorflow_gpu]

depending on whether you have an NVIDIA GPU available or not.

To get started, let's run Gymnos CLI to check if installation was successful. We will solve Boston Housing dataset `Boston Housing dataset <https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html>`_

.. code-block:: console

  $ gymnos train examples/experiments/boston_housing.json

Extra dependencies
-----------------------

To make Gymnos as light as possible and at the same time give flexibility to any developer to write their module with the library they feel most comfortable with, the extra dependencies will not be installed by default when installing Gymnos.

.. tip::

  Most components don't require extra dependencies. Check ``setup.py`` file to see if the component you want to use needs any extra dependency.

Several extra dependencies configurations have been setup so that you can choose the one that best suits your needs:

- ``pip3 install -e .[datasets]``: installs dependencies for all datasets
- ``pip3 install -e .[models]``: installs dependencies for all models
- ``pip3 install -e .[trackers]``: installs dependencies for all trackers
- ``pip3 install -e .[preprocessors]``: installs dependencies for all preprocessors

If you want to install Gymnos with all the dependencies:

.. code-block:: console

    $ pip3 install -e .[complete]

It is also possible to install dependencies for individual components using the following syntax:

- ``pip3 install -e .[datasets.<dataset_name>]``: instala las dependencias para dataset_name
- ``pip3 install -e .[models.<model_name>]``: instala las dependencias para model_name
- ``pip3 install -e .[trackers.<tracker_name>]``: instala las dependencias para tracker_name
- ``pip3 install -e .[preprocessors.<preprocessor_name>]``: instala las dependencias para preprocessor_name

.. note:: 

  By default, if a component needs an extra package and it is not installed, Gymnos will try to install it automatically. If you want to disable this feature, set the environment variable ``GYMNOS_AUTOINSTALL``.

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

.. code-block:: console

  $ git clone https://github.com/Telefonica/gymnos/tree/devel
  $ cd gymnos

Then build Docker container:

.. code-block:: console

  $ docker build -t gymnos .

If you are lucky enough to have a GPU, you just need to execute the following command to use the GPU in your Docker image.  

.. code-block:: console

  $ docker build -f Dockerfile.gpu -t gymnos-gpu .

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

.. code-block:: console

  $ docker login dockerhub.hi.inet

.. note::

  Please provide your corporate credentials for <USER_ID> and <USER_PASSWORD>

Then pull the latest image:

.. code-block:: console

  $ docker pull dockerhub.hi.inet/dcip/aura-prototypes/gymnos

or for gpu environments:

.. code-block:: console

  $ docker pull dockerhub.hi.inet/dcip/aura-prototypes/gymnos:gpu-latest

Now check that Docker images were indeed successfully pulled. You should see something like this

.. code-block:: console

  $ docker images  

  REPOSITORY                                                 TAG                 IMAGE ID            CREATED             SIZE
  dockerhub.hi.inet/dcip/aura-prototypes/gymnos              gpu-latest          4a55d3c18419        18 minutes ago      4.54GB
  dockerhub.hi.inet/dcip/aura-prototypes/gymnos              latest              37d2d2b9cd0a        18 minutes ago      2.54GB

Run image
-------------------

.. code-block:: console

  $ docker run -it gymnos


GPU version.

.. note::
  Please make sure `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_  is install in your computer.
  Refer to the following link for instructions about downloading and installing `nvidia-docker on Ubuntu 18.04 <https://cnvrg.io/how-to-setup-docker-and-nvidia-docker-2-0-on-ubuntu-18-04/>`_

.. code-block:: console

  $ nvidia-docker run -it gymnos-gpu

To get started let's run Gymnos CLI to check if installation was successful. We will solve Boston Housing dataset `Boston Housing dataset <https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html>`_

.. code-block:: console

  $ gymnos train examples/experiments/boston_housing.json
