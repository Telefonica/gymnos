.. index:: ! installing

.. _installing-gymnos:

################################
Installing the Gymnos environment
################################

Docker
==========

We provide up to date docker builds for the compiler. The ``stable``
repository contains released versions while the ``nightly``
repository contains potentially unstable changes in the develop branch.

.. code-block:: bash

    docker build -t gymnos-devel .


.. code-block:: bash

    docker run gymnos-devel:latest --version

You need to install the following dependencies:

+-----------------------------------+-------------------------------------------------------+
| Software                          | Notes                                                 |
+===================================+=======================================================+
| `Git for Linux`_                  | Command-line tool for retrieving source from Github.  |
+-----------------------------------+-------------------------------------------------------+
| `Visual Studio 2017 Build Tools`_ | C++ compiler                                          |
+-----------------------------------+-------------------------------------------------------+
| `Visual Studio 2017`_  (Optional) | C++ compiler and dev environment.                     |
+-----------------------------------+-------------------------------------------------------+

.. _Git for Linux: https://git-scm.com/download/linux
.. _Visual Studio 2017: https://www.visualstudio.com/vs/
.. _Visual Studio 2017 Build Tools: https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017

Example with bullets:

* Visual Studio C++ core features
* VC++ 2017 v141 toolset (x86,x64)
* Windows Universal CRT SDK
* Windows 8.1 SDK
* C++/CLI support


Clone the Repository
--------------------

To clone the source code, execute the following command:

.. code-block:: bash

    git clone --recursive https://github.com/ethereum/solidity.git
    cd solidity

If you want to help developing Solidity,
you should fork Solidity and add your personal fork as a second remote:

.. code-block:: bash

    git remote add personal git@github.com:[username]/solidity.git

External Dependencies
---------------------

We have a helper script which installs all required external dependencies
on macOS, Windows and on numerous Linux distros.

.. code-block:: bash

    ./scripts/install_deps.sh

Or, on Windows:

.. code-block:: bat

    scripts\install_deps.bat


Command-Line Build
------------------

**Be sure to install External Dependencies (see above) before build.**

Solidity project uses CMake to configure the build.
You might want to install ccache to speed up repeated builds.
CMake will pick it up automatically.
Building Solidity is quite similar on Linux, macOS and other Unices:

.. code-block:: bash

    mkdir build
    cd build
    cmake .. && make

or even easier:

.. code-block:: bash

    #note: this will install binaries solc and soltest at usr/local/bin
    ./scripts/build.sh

And for Windows:

.. code-block:: bash

    mkdir build
    cd build
    cmake -G "Visual Studio 15 2017 Win64" ..



Other example with bullets:

- the version number
- pre-release tag, usually set to ``develop.YYYY.MM.DD`` or ``nightly.YYYY.MM.DD``
- commit in the format of ``commit.GITHASH``
- platform, which has an arbitrary number of items, containing details about the platform and compiler

