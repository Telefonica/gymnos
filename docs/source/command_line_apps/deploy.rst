####################
Command: deploy
####################

Deploy saved trainer to SOFIA.

.. note::

    This command requires additional dependencies. Install gymnos with ``deploy`` dependencies to get started:

    .. code-block:: console

        $ pip3 install .[deploy]

Usage
================

.. argparse::
    :ref: scripts.cli.build_parser
    :prog: gymnos
    :path: deploy


Examples
================

These examples assume that the saved trainer file is called ``saved_trainer.zip``.

Deploy prompting for metadata (title, description, etc ...):

.. code-block:: console

    $ gymnos deploy saved_trainer.zip


Deploy with a JSON containing the metadata:

.. code-block:: console

    $ gymnos deploy saved_trainer.zip --metadata meta.json

