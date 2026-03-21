from trackey.core.registries.analyzer import ANALYZER_REGISTRY
from trackey.core.interfaces.analyzer import Analyzer


def build_analyzer(analyzer_type: str, **kwargs) -> Analyzer:
    cls = ANALYZER_REGISTRY.get(analyzer_type)
    if cls is None:
        raise ValueError(
            f"Unknown analyzer '{analyzer_type}'. "
            f"Available: {list(ANALYZER_REGISTRY.keys())}"
        )

    return cls(**kwargs)