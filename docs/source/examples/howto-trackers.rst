###################################
How to create a Tracker
###################################

Implementing a tracker in Gymnos is really simple, just inherit from ``Tracker`` and override some methods.

.. note::
    The training configuration (:class:`lib.core.tracker.Tracker`) will read ``lib.var.trackers.json`` to find the tracker given the tracker type. If you want to add a tracker, give it a name and add the location of the tracker.

.. note::
    No method is mandatory.

.. autoclass:: lib.trackers.tracker.Tracker
    :members:
