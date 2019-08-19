####################
Command: serve
####################

Run API REST server to make predictions.

.. note::

    This command requires additional dependencies. Install gymnos with ``serve`` dependencies to get started:

    .. code-block:: console

        $ pip3 install .[serve]

Usage
--------

.. argparse::
    :ref: scripts.cli.build_parser
    :prog: gymnos
    :path: serve


Examples
----------

These examples assume that the saved trainer file is called ``saved_trainer.zip``.

To listen to all interfaces:

.. code-block:: console

    $ gymnos serve saved_trainer.zip  --host 0.0.0.0


To listen to port 8080:

.. code-block:: console

    $ gymnos serve saved_trainer.zip --port 8080


To enable debug mode:

.. code-block:: console

    $ gymnos serve saved_trainer.zip --debug 1
