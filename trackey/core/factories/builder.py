import yaml
import logging
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Builder(ABC):
    # ------------------------------------------------------------------ #
    # Public Interface                                                   #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def build(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Shared functions                                                   #
    # ------------------------------------------------------------------ #
    def _load_yaml(self, cfg_path: str):
        cfg_path = Path(cfg_path)
        class_name = self.__class__.__name__

        if not cfg_path.exists():
            logger.error(f"[{class_name}] Config file not found: {cfg_path.resolve()}")
            raise FileNotFoundError(f"[{class_name}] Config file not found: {cfg_path.resolve()}")

        with cfg_path.open("r") as f:
            return yaml.safe_load(f)