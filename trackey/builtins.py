"""
Importing this module triggers the registration of all built-in plugins 
and components into their respective registries via import side-effects.
"""

import logging

logger = logging.getLogger(__name__)

def register_builtins():
    try:
        # IO Plugins
        from trackey.plugins.io.camera import CameraSourcePlugin
        from trackey.plugins.io.video import VideoSourcePlugin
        from trackey.plugins.io.display import DisplaySinkPlugin
        from trackey.plugins.io.rtsp import RtspSourcePlugin
        
        # Subscribers
        from trackey.plugins.subscribers.mail import MailSubscriberPlugin
        from trackey.plugins.subscribers.null import NullSubscriberPlugin
        from trackey.plugins.subscribers.webhook import WebhookSubscriberPlugin
        
        # Stores
        from trackey.plugins.stores.postgress_embedding import PostgresEmbeddingRepositoryPlugin
        from trackey.plugins.stores.postgres import PostgresRepositoryPlugin
        from trackey.plugins.stores.faiss import FaissPlugin
        
        # Core Components (Detectors, Trackers, Analyzers, etc)
        from trackey.core.io.output.viewer import OpenCVViewer
        from trackey.core.detectors import YoloDetector
        from trackey.core.trackers import DeepSortTracker
        from trackey.core.recognition.person.osnet import OsNetReid
        from trackey.core.analyzers import Counter
        from trackey.core.rendering.opencv_renderer import OpenCVRenderer
        from trackey.core.events.types import CountExceededEvent
        
        try:
            from trackey.core.recognition.face.retina import RetinaFaceDetector
        except ImportError:
            pass # Face recognition optional dependencies might not be installed
            
        logger.debug("Successfully registered all built-in Trackey plugins and components.")
    except Exception as e:
        logger.error(f"Failed to register some built-in components: {e}")
        raise

# Auto-register when imported
register_builtins()
