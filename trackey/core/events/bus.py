from typing import Dict, List, Type
from collections import defaultdict
from queue import Queue, Empty
from threading import Thread, Lock
import logging

from trackey.data.schemas.event             import BaseEvent
from trackey.core.interfaces.subscriber     import Subscriber


logger = logging.getLogger(__name__)

class EventBus:
    def __init__(self):
        self.event_queue: Queue                              = Queue()
        self.subscribers: Dict[Type[BaseEvent], List[Subscriber]] = defaultdict(list)  # actual dict
        self.lock:        Lock                               = Lock()
        self.running:     bool                               = True
        self.worker:      Thread                             = Thread(
            target=self._worker,
            daemon=True
        )
        self.worker.start()

    def subscribe(self, event_type: Type[BaseEvent], subscriber: Subscriber) -> None:
        self.subscribers[event_type].append(subscriber)

    def publish(self, event: BaseEvent) -> None:
        self.event_queue.put(event)
    
    def _worker(self):
        while self.running:
            try:
                event = self.event_queue.get(timeout=0.5)
                self._dispatch(event)
            except Empty:
                continue

    def _dispatch(self, event):
        event_subs: List[Subscriber] = self.subscribers[type(event)]
        for sub in event_subs:
            try:
                sub.on_event(event)
            except Exception as e:
                logger.error(f"[EventBus] Subscriber error: {e}")

    def stop(self):
        self.running = False
        self.worker.join()