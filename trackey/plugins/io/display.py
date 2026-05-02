from trackey.core.io.output.display import DisplaySink
from trackey.plugins.io.sink import SinkPlugin
from trackey.core.register import register_sink


@register_sink("display")
class DisplaySinkPlugin(SinkPlugin):

    @classmethod
    def validate(cls, cfg: dict) -> None:
        pass

    @classmethod
    def build(cls, cfg: dict) -> DisplaySink:
        cls.validate(cfg)
        params = cfg.get("params", {})
        return DisplaySink(
            window_name=params.get("window_name", "Trackey"),
            display_width=params.get("display_width"),
            display_height=params.get("display_height")
        )