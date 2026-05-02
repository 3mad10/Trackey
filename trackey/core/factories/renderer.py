import logging

from trackey.core.factories.builder     import Builder
from trackey.core.scene.scene           import Scene
from trackey.core.interfaces.renderer   import Renderer
from trackey.core.registries.render     import RENDERER_REGISTRY

logger = logging.getLogger(__name__)


class RendererBuilder(Builder):
    def __init__(self, cfg_path: str, scene: Scene):
        self.cfg = self._load_yaml(cfg_path)
        self.scene = scene
    
    def build(self) -> Renderer:
        renderer_cfg = self.cfg.get("renderer", {})

        if not isinstance(renderer_cfg, dict):
            raise ValueError("[RendererBuilder] 'renderer' must be a dict")

        renderer_type = renderer_cfg.get("type")
        if not renderer_type:
            raise ValueError("[RendererBuilder] Missing renderer type")

        params = renderer_cfg.get("params", {})

        renderer_cls = RENDERER_REGISTRY.get(renderer_type)
        if not renderer_cls:
            raise ValueError(f"[RendererBuilder] Unknown renderer '{renderer_type}'")

        return renderer_cls(scene=self.scene, **params)