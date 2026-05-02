from trackey.core.io.input.video import VideoFileSource
from trackey.plugins.io.source import SourcePlugin
from trackey.core.register import register_source


@register_source("video")
class VideoSourcePlugin(SourcePlugin):

    @classmethod
    def validate(cls, cfg: dict) -> None:
        params = cfg.get("params", {})
        if "path" not in params:
            raise ValueError(
                "[VideoSourcePlugin] Missing required param 'path'.\n"
                "source:\n"
                "  type: video\n"
                "  params:\n"
                "    path: input.mp4\n"
            )

    @classmethod
    def build(cls, cfg: dict) -> VideoFileSource:
        cls.validate(cfg)
        params = cfg.get("params", {})
        return VideoFileSource(
            path=params["path"],
            width=params.get("width"),
            height=params.get("height"),
            target_fps=params.get("target_fps"),
            drop_strategy=params.get("drop_strategy", "block")
        )