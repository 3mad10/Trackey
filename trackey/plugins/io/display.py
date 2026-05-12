from trackey.plugins.io.sink import SinkPlugin
from trackey.core.register import register_sink
from trackey.core.io.output.viewer.opencv_viewer import OpenCVViewer


@register_sink("display")
class DisplaySinkPlugin(SinkPlugin):

    @classmethod
    def validate(cls, cfg: dict) -> None:
        pass

    @classmethod
    def build(cls, cfg: dict) -> OpenCVViewer:
        params = cfg.get("params", {})
        return OpenCVViewer(
            window_name=params.get("window_name", "Trackey")
        )