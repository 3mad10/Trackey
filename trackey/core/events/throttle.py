import time
from typing import Dict

from trackey.data.schemas.event import BaseEvent
from trackey.core.interfaces.subscriber import Subscriber


class ThrottledSubscriber(Subscriber):
    def __init__(self, subscriber: Subscriber,
                 throttle_seconds: float):
        self._subscriber      = subscriber
        self.throttle_seconds = throttle_seconds
        self._last_fired: Dict = {}

    def on_event(self, event: BaseEvent) -> None:
        if not self._should_fire(event):
            return
        self._subscriber.on_event(event)
        self._last_fired[(type(event), event.camera_id)] = time.monotonic()
    
    def _should_fire(self, event: BaseEvent) -> bool:
        if self.throttle_seconds <= 0:
            return True
        key  = (type(event), event.camera_id)
        last = self._last_fired.get(key)
        if last is None:
            return True
        return (time.monotonic() - last) >= self.throttle_seconds