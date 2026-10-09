from trackey.core.events.subscribers.webhook import WebhookSubscriber
from trackey.plugins.subscribers.subscriber import SubscriberPlugin
from trackey.core.register import register_subscriber

@register_subscriber("webhook")
class WebhookSubscriberPlugin(SubscriberPlugin):

    @classmethod
    def validate(cls, cfg: dict) -> None:
        params = cfg.get("params", {})
        if "url" not in params:
            raise ValueError(
                "[WebhookSubscriberPlugin] Missing required param 'url'.\n"
            )

    @classmethod
    def build(cls, cfg: dict) -> WebhookSubscriber:
        cls.validate(cfg)
        params = cfg.get("params", {})
        return WebhookSubscriber(
            url=params["url"],
            timeout=params.get("timeout", 2.0),
            headers=params.get("headers", None)
        )
