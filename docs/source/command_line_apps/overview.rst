##################
Command-Line Apps
##################

Once you have installed Gymnos in your system, you will have several applications available on your command line that will make it possible to train a complete supervised learning system without touching a single line of code, obtain predictions directly from the command line or create a prediction server in a very simple way.

Usage
*******
.. code-block:: console

    $ gymnos -h

        usage: gymnos [-h] {train,predict,serve} ...

        Gymnos tool

        positional arguments:
          {train,predict,serve}

        optional arguments:
          -h, --help            show this help message and exit

All Command-Line Apps
**********************

.. toctree::
    :maxdepth: 1

    gymnos train <train>
    gymnos predict <predict>
    gymnos serve <serve>
    gymnos deploy <deploy>

