#
#
#   Test history tracker
#
#

import pytest

from gymnos.trackers import History


@pytest.fixture
def history():
    tracker = History()
    tracker.start()
    return tracker


class TestHistoryTracker:

    def test_add_tag(self, history):
        history.add_tag("hello")
        history.add_tag("world")
        assert history.tags == ["hello", "world"]

    def test_add_tags(self, history):
        history.add_tags(["hello", "world"])
        assert history.tags == ["hello", "world"]

    def test_log_asset(self, history):
        history.log_asset("image", "folder/image.png")
        history.log_asset("document", "folder/document.pdf")
        assert "document" in history.assets and history.assets["document"] == "folder/document.pdf"
        assert "image" in history.assets and history.assets["image"] == "folder/image.png"

    def test_log_image(self, history):
        history.log_image("image", "folder/image.png")
        history.log_image("image2", "folder/image2.png")

        assert "image" in history.images and history.images["image"] == "folder/image.png"
        assert "image2" in history.images and history.images["image2"] == "folder/image2.png"

    def test_log_figure(self, history):
        ...

    def test_log_metric(self, history):
        ...

    def test_log_metrics(self, history):
        ...

    def test_log_param(self, history):
        ...

    def test_log_params(self, history):
        ...
