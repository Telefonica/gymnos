#####################
Command: predict
#####################

Predict samples using saved trainer.

Usage
-------------
.. argparse::
    :ref: scripts.cli.build_parser
    :prog: gymnos
    :path: predict

Examples
-------------

These examples assume that the saved trainer file is called ``saved_trainer.zip``.

To predict an image:

.. code-block:: console

    $ gymnos predict saved_trainer.zip --image dog.png
        {
            "predictions": [
                0
            ],
            "probabilities": [
                [0.75, 0.2]
            ],
            "classes": {
                "names": [
                    "dog",
                    "cat"
                ],
                "total": 2
            }
        }

To predict multiple images:

.. code-block:: console

    $ gymnos predict saved_trainer.zip --image dog.png cat.png
        {
            "predictions": [
                0,
                1
            ],
            "probabilities": [
                [0.75, 0.2],
                [0.1,  0.9]
            ],
            "classes": {
                "names": [
                    "dog",
                    "cat"
                ],
                "total": 2
            }
        }

To predict a dataframe:

.. code-block:: json
    :caption: samples.json

    [
        [12.3, 4.5, 6.4],
        [1.45, 2.3, 10.9]
    ]


.. code-block:: console

    $ gymnos predict saved_trainer.zip --json samples.json

        {
            "predictions": [
                13.5,
                6.7
            ]
        }

To predict text:

.. code-block:: json
    :caption: samples.json

    [
        "this is an angry text",
        "this is a happy text",
        "this is a super angry text"
    ]


.. code-block:: console

    $ gymnos predict saved_trainer.zip --json samples.json

        {
            "predictions": [
                0,
                1,
                0
            ],
            "probabilities": [
                [0.8, 0.2],
                [0.4, 0.6],
                [0.9, 0.1]
            ],
            "classes": {
                "names": [
                    "negative",
                    "positive"
                ],
                "total": 2
            }
        }

.. tip::

    You can save predictions to a JSON file by using the following command:

    .. code-block:: console

        $ gymnos predict saved_trainer.zip --json samples.json > predictions.json 
