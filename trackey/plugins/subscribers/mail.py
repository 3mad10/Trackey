import os

from trackey.core.events.subscribers.mail import MailSubscriber
from trackey.plugins.subscribers.subscriber import SubscriberPlugin
from trackey.core.register import register_subscriber


@register_subscriber("mail")
class MailSubscriberPlugin(SubscriberPlugin):

    @classmethod
    def validate(cls, cfg: dict) -> None:
        params = cfg.get("params", {})
        if "to" not in params:
            raise ValueError(
                "[MailSubscriberPlugin] Missing required param 'to'.\n"
                "params:\n"
                "  to: [ops@company.com]\n"
                "  sender: trackey@company.com\n"
            )
        if not isinstance(params["to"], list):
            raise ValueError(
                "[MailSubscriberPlugin] 'to' must be a list"
            )
        if "sender" not in params:
            raise ValueError(
                "[MailSubscriberPlugin] Missing required param 'sender'"
            )
        if "password" not in params:
            raise ValueError(
                "[MailSubscriberPlugin] Missing 'password'.\n"
                "Set the environment variable and reference it:\n"
                "params:\n"
                "  password: ${TRACKEY_SMTP_PASSWORD}\n"
            )

    @classmethod
    def build(cls, cfg: dict) -> MailSubscriber:
        cls.validate(cfg)
        params = cfg.get("params", {})
        password = params.get("password")
        return MailSubscriber(
            to=params["to"],
            sender=params["sender"],
            smtp_host=params.get("smtp_host", "smtp.gmail.com"),
            smtp_port=params.get("smtp_port", 587),
            password=password
        )
