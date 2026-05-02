import logging
from pathlib import Path
from trackey.core.factories.builder import Builder

logger = logging.getLogger(__name__)


class SinkBuilder(Builder):
    def __init__(self, cfg_path: str):
        self.cfg = self._load_yaml(cfg_path)
    