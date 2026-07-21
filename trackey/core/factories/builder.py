import re
import os
import yaml
import logging
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

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

        with open(cfg_path) as f:
            raw = f.read()
        resolved = self._resolve_env_vars(raw)
        return yaml.safe_load(resolved)
    
    def _resolve_env_vars(self, text: str) -> str:
        def replace(match):
            var_name = match.group(1)
            value = os.environ.get(var_name)
            if value is None:
                raise ValueError(
                    f"[Builder] Config references environment variable "
                    f"'{var_name}' which is not set."
                )
            return value
        return _ENV_VAR_PATTERN.sub(replace, text)