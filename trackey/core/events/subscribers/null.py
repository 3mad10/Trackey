from trackey.core.interfaces.subscriber import Subscriber
from trackey.data.schemas.event import BaseEvent



class NullSubscriber(Subscriber):
    def __init__(self):
        pass

    def on_event(self, event: BaseEvent) -> None:
        print(f"Event Occured : {event}")
