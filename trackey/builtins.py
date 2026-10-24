"""
Importing this module triggers the registration of all built-in plugins
and components into their respective registries via import side-effects.
"""

import logging

logger = logging.getLogger(__name__)


def register_builtins():
    try:
        # IO Plugins
        import trackey.plugins.io.camera  # noqa: F401
        import trackey.plugins.io.video  # noqa: F401
        import trackey.plugins.io.display  # noqa: F401

        # Subscribers
        import trackey.plugins.subscribers.mail  # noqa: F401
        import trackey.plugins.subscribers.null  # noqa: F401
        import trackey.plugins.subscribers.webhook  # noqa: F401

        # Stores
        import trackey.plugins.stores.postgress_embedding  # noqa: F401
        import trackey.plugins.stores.postgres  # noqa: F401
        import trackey.plugins.stores.faiss  # noqa: F401

        # Core Components (Detectors, Trackers, Analyzers, Renderers, Events)
        import trackey.core.io.output.viewer.opencv_viewer  # noqa: F401
        import trackey.core.detectors.yolo  # noqa: F401
        import trackey.core.trackers.deepsort  # noqa: F401
        import trackey.core.recognition.person.osnet  # noqa: F401
        import trackey.core.analyzers.counter  # noqa: F401
        import trackey.core.rendering.opencv_renderer  # noqa: F401
        import trackey.core.events.types  # noqa: F401

        try:
            import trackey.core.recognition.face.retina  # noqa: F401
        except ImportError:
            pass  # Optional dependency

        logger.debug("Successfully registered all built-in Trackey plugins and components.")
    except Exception as e:
        logger.error(f"Failed to register some built-in components: {e}")
        raise


# Auto-register when imported
register_builtins()
