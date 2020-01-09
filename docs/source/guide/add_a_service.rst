####################
Add a service
####################

Overview
===============
Gymnos services provides a way to download files from services in a simple way.

To enable this, each service implements a subclass of :class:`gymnos.services.service.Service`, which specifies:

* How to download files
* Configuration variables for service

Writing ``my_service.py``
=============================

Use the default template
--------------------------
If you want to :ref:`contribute to our repo <contributing>` and add a new service, the following script will help you get started generating the required python files. To use it, clone the `Gymnos <https://github.com/Telefonica/gymnos>`_ repository and run the following command:

.. code-block:: console

  $ python3 -m scripts.create_new service --name my_service

This command will create ``gymnos/services/my_service.py``, and modify ``gymnos/__init__.py`` to register service so we can load it using ``gymnos.load``.

The service registration process is done by associating the service id with their path:

.. code-block:: python
    :caption: gymnos/__init__.py

    services.register(
        type="my_service",
        entry_point="gymnos.services.my_service.MyService"
    )

Go to ``gymnos/services/my_service.py`` and then search for TODO(my_service) in the generated file to do the modifications.

Service
-----------
Each service is defined as a subclass of :class:`gymnos.services.service.Service` implementing any methods needed to download files and overriding ``Config`` class variable to define the configuration variables for the service.

my_service.py
----------------

``my_service.py`` first look like this:

.. code-block:: python

    #
    #
    #   MyService
    #
    #

    from .. import config
    from .service import Service

    class MyService(Service):
        """
        TODO(my_service): Description of my service.
        """

        class Config(config.Config):
            """
            OPTIONAL(my_service): Define your required and optional configuration variables.
            """

        def download(self, *args, **kwargs):
            """
            TODO(my_service): Download file.
            """

Adding download methods
==========================

Add a method that download file(s) with any parameters you need:

.. code-block::

    def download(self, dataset_name=None, competition_name=None, verbose=True):
        dataset_downloader.download(dataset_name, verbose=verbose)

By convention, the public method to download files is ``download`` but we don't restrict the methods to implement. This is also valid:

.. code-block::

    def download_dataset(self, name, verbose=True):
        ...

    def download_competition(self, name, verbose=True):
        ...

Any service inherits from :class:`gymnos.services.service.Service` so you have available the following attributes from constructor:

- ``self.download_dir``: Directory to download files
- ``self.force_download``: Whether or not force download if file exists.
- ``self.config``: Values for configuration variables.

.. note::

    If the file to download already exists, a download is not needed so you should return the path from downloads directory unless ``self.force_download`` is ``True``.
    Note that if your service requires authentication, you should always authenticate user.

Specifying ``Config``
========================================

Use the ``Config`` class to define your required and optional variables. The user will specify these values using environment variables or using 
a configuration located at ~/.gymnos/gymnos.json.

.. code-block:: python

    class MyService(Service):

        class Config(config.Config):

            MY_SERVICE_SECRET_KEY = config.Value(required=True, help="Secret Key for my_service.com")  # required variable
            MY_SERVICE_PROXY = config.Value(default="proxy.com", help="Proxy for my_service.com")  # optional variable with default
            MY_SERVICE_TIME = config.Value(default=lambda: datetime.now(), help="Current time")  # optional variable with callable default

Then, you can use the values for this variables in your methods using ``self.config``. If the user has not provided all required variable, an exception is thrown:

.. code-block:: python

    def download(self, *args, **kwargs):
        download_from_my_service(secret_key=self.config.MY_SERVICE_SECRET_KEY)


Summary
==============

1. Create ``MyService`` in ``gymnos/services/my_service.py`` inheriting from :class:`gymnos.services.service.Service`
2. Define configuration variables overriding class variable ``Config``
3. Implement methods to download files

Adding the dataset to ``Telefonica/gymnos``
=============================================

If you'd like to share your work with the community, you can check in your service implementation to Telefonica/gymnos. Thanks for thinking of contributing!

Before you send your pull request, follow these last few steps (check :ref:`contributing` to see more details):

1. Test service with any Gymnos dataset
-----------------------------------------------------
Check that your service is working with a Gymnos dataset.

2. Add documentation
----------------------
Add service documentation.

3. Check your code style
--------------------------
Follow the `PEP8 Python style guide <https://www.python.org/dev/peps/pep-0008/>`_, except Gymnos uses 120 characters as maximum line length.

You can lint files running ``flake8`` command:

.. code-block:: console

    $ flake8

Adding the service from other repository
=================================================

You can also add a service from other repository in a very simple way by converting your repository into a Python library.

Once you have defined your ``setup.py``, create and register your Gymnos services in the same way we have shown.

Here is a minimal example. Say we have our library named ``gymnos_my_services`` and we want to add the service ``my_service``. You have to:

1. Create ``MyService`` in ``gymnos_my_services/my_service.py`` inheriting from :class:`gymnos.services.service.Service` and implementing the abstract methods
2. Register service in your module ``__init__.py`` referencing the type and the path:

.. code-block:: python
    :caption: gymnos_my_services/__init__.py

    import gymnos

    gymnos.services.register(
        type="my_service",
        entry_point="gymnos_my_services.my_service.MyService"
    )


That's it, when someone wants to run ``my_service`` from ``gymnos_my_services``, simply ``pip install`` the package and reference the package when you are loading the service with the following format: ``<module_name>:<service_name>``.

For example:

.. code-block:: python

    gymnos.services.load("gymnos_my_services:my_service")
