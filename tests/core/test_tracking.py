#
#
#   Test tracking
#
#

import pytest
import gymnos

from gymnos.core.tracking import Tracking


def test_tracker_instance():
    tracker_1 = dict(
        type="tensorboard"
    )
    tracker_2 = dict(
        type="mlflow"
    )
    tracking = Tracking(trackers=[tracker_1, tracker_2])

    assert isinstance(tracking.trackers, gymnos.trackers.tracker.TrackerList)

    assert len(tracking.trackers) == 2

    tracking = Tracking()

    assert isinstance(tracking.trackers, gymnos.trackers.tracker.TrackerList)

    assert len(tracking.trackers) == 0

    tracker = dict(
        type="dummy"
    )

    with pytest.raises(ValueError):
        Tracking(trackers=[tracker])
