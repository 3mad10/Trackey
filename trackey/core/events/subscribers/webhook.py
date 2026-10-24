import requests
import logging
from dataclasses import asdict

from trackey.core.interfaces.subscriber import Subscriber
from trackey.data.schemas.event import BaseEvent

logger = logging.getLogger(__name__)

class WebhookSubscriber(Subscriber):
    def __init__(self, url: str, timeout: float = 2.0, headers: dict = None):
        self.url = url
        self.timeout = timeout
        self.headers = headers or {"Content-Type": "application/json"}
        self.session = requests.Session()

    def on_event(self, event: BaseEvent) -> None:
        try:
            payload = asdict(event)
            payload["event_type"] = event.__class__.__name__

            def _serialize(obj):
                if isinstance(obj, dict):
                    return {k: _serialize(v) for k, v in obj.items() if k != "frame"}
                elif isinstance(obj, list):
                    return [_serialize(item) for item in obj]
                elif hasattr(obj, "isoformat"):
                    return obj.isoformat()
                elif hasattr(obj, "hex"):
                    return str(obj)
                return obj

            payload = _serialize(payload)

            resp = self.session.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            resp.raise_for_status()
            logger.debug(f"[WebhookSubscriber] Successfully sent event {event.__class__.__name__} to {self.url}")
        except Exception as e:
            logger.error(f"[WebhookSubscriber] Failed to send event to {self.url}: {e}")
