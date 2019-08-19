import gymnos
import pytest


def test_load():
    tracker = gymnos.trackers.load("tensorboard")

    assert isinstance(tracker, gymnos.trackers.tracker.Tracker)

    with pytest.raises(ValueError):
        _ = gymnos.trackers.load("dummy")
