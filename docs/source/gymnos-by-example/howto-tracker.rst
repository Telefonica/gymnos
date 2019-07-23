###################################
How to create a Tracker
###################################

Implementing a tracker in Gymnos is really simple, just inherit from ``Tracker`` and override some methods.

.. note::
    The training configuration (:class:`gymnos.core.tracker.Tracker`) will read ``gymnos.var.trackers.json`` to find the tracker given the tracker's type. If you want to add a tracker, give it a name and add the tracker's location.

.. note::
    No method is mandatory.

.. autoclass:: gymnos.trackers.tracker.Tracker
    :members:
