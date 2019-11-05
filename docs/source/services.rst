########################
Services
########################

Gymnos services are a collection of packages to download from different services like Kaggle, SAMBA or HTTP.

Some services require configuration variables, e.g credentials. You can set these variables in two ways:

- Environment variables
- JSON file (~/.gymnos/gymnos.json)

For example, if a service requires two variables named ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` you can set them with environment variables, e.g:

.. code-block:: console

    $ export KAGGLE_USERNAME=xxxxxxxxxxxxxxxxx
    $ export KAGGLE_KEY=xxxxxxxxxxxxxxxxx

Or writing them to json file in ~/.gymnos/gymnos.json:

.. code-block:: json
    :caption: ~/.gymnos/gymnos.json

    {
        "KAGGLE_USERNAME": "xxxxxxxxxxxxxxxxx",
        "KAGGLE_KEY": "xxxxxxxxxxxxxxxxx"
    }

Usage
***********

.. code-block:: python

    kaggle = gymnos.services.load("kaggle", download_dir="downloads", config_files=["credentials.json"])  # look for configuration variables in credentials.json

    kaggle.download(dataset_name="mlg-ulb/creditcardfraud", files=None)  # download full dataset to download_dir

You can use them in combination with :class:`gymnos.services.download_manager.DownloadManager`:

.. code-block:: python

    dl_manager = DownloadManager(download_dir="downloads")
    file_paths = dl_manager["http"].download({  # download HTTP files and return dict with their paths
        "file_1": "http://example.com/1",
        "file_2": "http://anotherexample.com/2"
    })

    file_extracted_paths = dl_manager.extract(file_paths)  # returns a dict with their extracted paths

    file_paths = dl_manager["smb"].download([  # download SAMBA files and return list their paths
        "smb://192.168.1.1/homes/johndoe/image.png",
        "smb://192.168.1.1/homes/johndoe/data.zip",
    ])

All Services
****************

.. contents:: 
    :local: 

http
========================
.. autoclass:: gymnos.services.http.HTTP
    :noindex:
    :members:

smb
========================
.. autoclass:: gymnos.services.smb.SMB
    :noindex:
    :members: download

    .. autoclass:: gymnos.services.smb.SMB.Config
        :noindex:

kaggle
========================
.. autoclass:: gymnos.services.kaggle.Kaggle
    :noindex:
    :members: download

    .. autoclass:: gymnos.services.kaggle.Kaggle.Config
        :noindex:

SOFIA
=============
.. autoclass:: gymnos.services.sofia.SOFIA
    :noindex:
    :members: download

    .. autoclass:: gymnos.services.sofia.SOFIA.Config
        :noindex:
