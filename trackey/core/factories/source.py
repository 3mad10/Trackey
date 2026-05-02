import logging

from trackey.core.factories.builder import Builder
from trackey.core.interfaces.source import InputSource
from trackey.core.registries.source import SOURCE_REGISTRY

logger = logging.getLogger(__name__)


class SourceBuilder(Builder):
    SOURCE_CFG_FORMAT = (
        "source:\n"
        "  type: <source-type>\n"
        "  params\n"
        "    <source-specific-params>\n"
        "\n"
        "video params:         path: <input-video-path>\n"
        "                      [Optional]width: <input-resize-width>\n"
        "                      [Optional]height: <input-resize-height>\n"
        "rtsp params:          url: <rtsp-url>\n"
        "                      [Optional]width: <input-resize-width>\n"
        "                      [Optional]height: <input-resize-height>\n"
        "camera params:        index: <index-of-webcam>\n"
        "                      [Optional]width: <input-resize-width>\n"
        "                      [Optional]height: <input-resize-height>\n"
    )
    def __init__(self, cfg_path: str):
        self.cfg = self._load_yaml(cfg_path)
    
    def build(self) -> InputSource:
        build_source = self._build_source()
        return build_source
    
    # ------------------------------------------------------------------ #
    # Build                                                              #
    # ------------------------------------------------------------------ #
    def _build_source(self):
        source_cfg = self.cfg.get("source", {})
        self._validate_source(source_cfg)
        source_type = source_cfg["type"]
        plugin_cls = SOURCE_REGISTRY.get(source_type)

        if not plugin_cls:

            raise ValueError(
                f"[SourceBuilder] Unsupported source type: {source_type}. "
                f"Available: {list(SOURCE_REGISTRY.keys())}"
            )

        plugin_cls.validate(source_cfg)
        return plugin_cls.build(source_cfg)

    # ------------------------------------------------------------------ #
    # Validation                                                         #
    # ------------------------------------------------------------------ #
    def _validate_source(self, source_cfg):
        self._validate_structure(source_cfg)
        self._validate_required_fields(source_cfg)
        
        
    def _validate_structure(self, source_cfg):
        if not isinstance(source_cfg, dict):
            msg = (
                f"[SourceBuilder] Source must be of type dict.\n"
                f"{self.SOURCE_CFG_FORMAT}"
            )
            logger.error(msg)
            raise ValueError(msg)
    
    def _validate_required_fields(self, source_cfg):
        if "type" not in source_cfg:
            raise ValueError(
                f"[SourceBuilder] Source missing type.\n"
                f"{self.SOURCE_CFG_FORMAT}"
            )
    