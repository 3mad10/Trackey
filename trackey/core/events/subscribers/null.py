from datetime import datetime
from trackey.core.interfaces.subscriber import Subscriber
from trackey.data.schemas.event import BaseEvent


class NullSubscriber(Subscriber):
    def __init__(self):
        pass

    def on_event(self, event: BaseEvent) -> None:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"Event Occured : {event}")
        print(f"Current Time : {current_time}")
