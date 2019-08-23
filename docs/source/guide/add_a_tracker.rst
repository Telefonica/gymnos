#################
Add a tracker
#################

Overview
=============

Machine Learning trackers are distributed in all kinds of API specifications and in all kinds of places, and they're not always implemented in a format that's ready to feed into a machine learning pipeline. Enter Gymnos trackers.

Gymnos trackers provides a way to transform all those trackers into a standard format to make them ready for a machine learning pipeline.

To enable this, each model implements a subclass of :class:`gymnos.trackers.Tracker`, which specifies:

* How to log a metric
* How to log a parameter
* How to log an image
* How to log a tag

Writing ``my_tracker.py``
==========================

Use the default template
-------------------------
If you want to :ref:`contribute to our repo <contributing>` and add a new tracker, the following script will help you get started generating the required python files. To use it, clone the `Gymnos <https://github.com/Telefonica/gymnos>`_ repository and run the following command:

.. code-block:: console

    $ python3 -m scripts.create_new tracker --name my_tracker

This command will create ``gymnos/trackers/my_tracker.py``, and modify ``gymnos/__init__.py`` to register tracker so we can load it using ``gymnos.load``.

The tracker registration process is done by associating the tracker name with their path:

.. code-block:: python
    :caption: gymnos/__init__.py

    trackers.register(
        name="my_tracker",
        entry_point="gymnos.trackers.my_tracker.MyTracker"
    )

Go to ``gymnos/trackers/my_tracker.py`` and then search for TODO(my_tracker) in the generated file to do the modifications.

Tracker
-----------
Each tracker is defined as a subclass of :class:`gymnos.trackers.Tracker` implementing the following methods:

* ``start``: initialize tracker
* ``end``: end tracker
* ``log_tag``: log tag
* ``log_asset``: log any file
* ``log_image``: log image
* ``log_figure``: log Matplotlib figure
* ``log_metric``: log metric
* ``log_param``: log parameter

my_tracker.py
----------------

.. code-block:: python

    #
    #
    #   MyTracker
    #
    #

    from .tracker import Tracker

    class MyTracker(Tracker):
        """
        TODO(my_tracker): Description of my tracker
        """
        def __init__(self, **parameters):
            # TODO(my_tracker): Define and initialize tracker parameters
        
        def start(self, run_id, logdir):
            # OPTIONAL: Initialize tracker
            pass
        
        def add_tag(self, tag):
            # OPTIONAL: Add tag
            pass
        
        def log_asset(self, name, file_path):
            # OPTIONAL: Log asset
            pass
        
        def log_image(self, name, file_path):
            # OPTIONAL: Log image
            pass
        
        def log_figure(self, name, figure):
            # OPTIONAL: Log Matplotlib figure
            pass
        
        def log_metric(self, name, value, step=None):
            # OPTIONAL: Log metric
            pass
        
        def log_param(self, name, value, step=None):
            # OPTIONAL: Log parameter
            pass
        
        def end(self):
            # OPTIONAL: Called when the experiment is finished
            pass


Summary
=============

1. Create ``MyTracker`` in ``gymnos/tracker/my_tracker.py`` inheriting from :class:`gymnos.trackers.tracker.Tracker` implementing any of the following available methods:

- ``start(run_id, logdir)``
- ``add_tag(self, tag)``
- ``log_asset(self, name, file_path)``
- ``log_image(self, name, file_path)``
- ``log_figure(self, name, figure)``
- ``log_metric(self, name, value, step=None)``
- ``log_param(self, name, value, step=None)``
- ``end(self)``

2. Register the tracker in ``gymnos/__init__.py`` by adding:

.. code-block:: python

    trackers.register(
        name="my_tracker",
        entry_point="gymnos.trackers.my_tracker.MyTracker"
    )


Adding the tracker to ``Telefonica/gymnos``
===========================================

If you'd like to share your work with the community, you can check in your tracker implementation to Telefonica/gymnos. Thanks for thinking of contributing!

Before you send your pull request, follow these last few steps (check :ref:`contributing` to see more details):


1. Add documentation
----------------------
Add tracker documentation.

2. Check your code style
--------------------------
Follow the `PEP8 Python style guide <https://www.python.org/dev/peps/pep-0008/>`_, except Gymnos uses 120 characters as maximum line length.

You can lint files running ``flake8`` command:

.. code-block:: console

    $ flake8

Adding the tracker from other repository
=================================================

You can also add a tracker from other repository in a very simple way by converting your repository in a Python library.

Once you have defined your ``setup.py``, create and register your Gymnos trackers in the same way we have shown.

Here is a minimal example. Say we have our library named ``gymnos_my_trackers`` and we want to add the tracker ``my_tracker``. You have to:

1. Create ``MyTracker`` in ``gymnos_my_trackers/my_tracker.py`` inheriting from :class:`gymnos.trackers.tracker.Tracker` and implementing the abstract methods
2. Register tracker in your module ``__init__.py`` referencing the name and the path:

.. code-block:: python
    :caption: gymnos_my_trackers/__init__.py

    import gymnos

    gymnos.trackers.register(
        name="my_tracker",
        entry_point="gymnos_my_trackers.my_tracker.MyTracker"
    )


That's it, when someone wants to run ``my_tracker`` from ``gymnos_my_trackers``, simply ``pip install`` the package and reference the package when you are loading the tracker with the following format: ``<module_name>:<tracker_name>``.

For example:

.. code-block:: python

    gymnos.trackers.load("gymnos_my_trackers:my_tracker")
