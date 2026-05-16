from trackey.core.io.input.camera import CameraSource
from trackey.plugins.io.source import SourcePlugin
from trackey.core.register import register_source


@register_source("camera")
class CameraSourcePlugin(SourcePlugin):
    @classmethod
    def validate(cls, cfg: dict):
        params = cfg.get("params", {})
        if "index" not in params:
            raise ValueError("[CameraSourcePlugin] Missing 'index'.")

    @classmethod
    def build(cls, cfg: dict):
        params = cfg["params"]
        return CameraSource(
            index=params["index"],
            width=params.get("width"),
            height=params.get("height")
        )