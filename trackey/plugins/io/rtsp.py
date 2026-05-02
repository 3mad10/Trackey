from trackey.core.io.input.rtsp import RtspSource
from trackey.plugins.io.source import SourcePlugin
from trackey.core.register import register_source


@register_source("rtsp")
class RtspSourcePlugin(SourcePlugin):
    @classmethod
    def validate(cls, cfg: dict):
        params = cfg.get("params", {})
        if "url" not in params:
            raise ValueError("[RtspSourcePlugin] Missing 'url'.")

    @classmethod
    def build(cls, cfg: dict):
        params = cfg["params"]
        return RtspSource(
            url=params["url"],
            width=params.get("width"),
            height=params.get("height")
        )