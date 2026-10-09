import requests
import logging
import json
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
            # We assume event is a dataclass
            payload = asdict(event)
            # Add the event type name for routing on receiver side
            payload["event_type"] = event.__class__.__name__

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
