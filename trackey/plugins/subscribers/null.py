import os

from trackey.core.events.subscribers.null import NullSubscriber
from trackey.plugins.subscribers.subscriber import SubscriberPlugin
from trackey.core.register import register_subscriber


@register_subscriber("null_subscriber")
class NullSubscriberPlugin(SubscriberPlugin):

    @classmethod
    def validate(cls, cfg: dict) -> None:
        pass

    @classmethod
    def build(cls, cfg: dict) -> NullSubscriber:
        return NullSubscriber()
