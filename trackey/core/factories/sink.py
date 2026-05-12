import logging
from typing import List
from trackey.core.factories.builder import Builder
from trackey.core.registries.sink import SINK_REGISTRY
from trackey.core.interfaces.sink import OutputSink

logger = logging.getLogger(__name__)


class SinkBuilder(Builder):

    SINKS_STRUCTURE = (
        "sinks:\n"
        "  - type: <sink-type>\n"
        "    params:\n"
        "      param1: value\n"
    )

    def __init__(self, cfg_path: str):
        self.cfg = self._load_yaml(cfg_path)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def build(self) -> List[OutputSink]:
        return self._build_sinks()

    # ------------------------------------------------------------------ #
    # Build                                                                #
    # ------------------------------------------------------------------ #

    def _build_sinks(self) -> List[OutputSink]:
        sinks_cfg = self.cfg.get("sinks", [])

        if not isinstance(sinks_cfg, list):
            raise TypeError(
                f"[SinkBuilder] sinks must be a list.\n"
                f"{self.SINKS_STRUCTURE}"
            )

        sinks = []
        for sink_cfg in sinks_cfg:
            self._validate_sink_cfg(sink_cfg)
            sink = self._build_sink(sink_cfg)
            sinks.append(sink)
            logger.info(
                f"[SinkBuilder] Built sink: {sink_cfg['type']}"
            )

        return sinks

    def _build_sink(self, sink_cfg: dict) -> OutputSink:
        sink_type   = sink_cfg["type"]
        plugin_cls  = SINK_REGISTRY.get(sink_type)

        if not plugin_cls:
            raise ValueError(
                f"[SinkBuilder] Unknown sink type: '{sink_type}'. "
                f"Available: {list(SINK_REGISTRY.keys())}"
            )

        # plugin handles validation + construction
        plugin_cls.validate(sink_cfg)
        return plugin_cls.build(sink_cfg)

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def _validate_sink_cfg(self, sink_cfg: dict) -> None:
        if not isinstance(sink_cfg, dict):
            raise TypeError(
                f"[SinkBuilder] Each sink must be a dict.\n"
                f"{self.SINKS_STRUCTURE}"
            )
        if "type" not in sink_cfg:
            raise ValueError(
                f"[SinkBuilder] Sink missing 'type'.\n"
                f"{self.SINKS_STRUCTURE}"
            )