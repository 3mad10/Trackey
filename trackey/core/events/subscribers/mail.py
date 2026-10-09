import smtplib
import os
import logging
import numpy as np


from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID
from typing import List, Optional

from trackey.core.interfaces.subscriber import Subscriber
from trackey.core.events.mails.message import MailMessage
from trackey.data.schemas.event import BaseEvent
from email.mime.text import MIMEText

from trackey.data.schemas.frame import Frame


logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

class MailSubscriber(Subscriber):
    def __init__(self,
                 to:        List[str],
                 sender:    str,
                 smtp_host: str = "smtp.gmail.com",
                 smtp_port: int = 587,
                 password:  Optional[str] = None):
        self.to        = to
        self.sender    = sender
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.password  = password

    def on_event(self, event: BaseEvent) -> None:
        try:
            message = self._format(event)
            self._send(message)
        except Exception:
            logger.exception(
                "[MailSubscriber] Failed to send mail notification"
            )

    def _format(self, event: BaseEvent) -> MailMessage:
        return MailMessage(
            sender=self.sender,
            to=self.to,
            subject=f"[Trackey] {type(event).__name__}",
            body=self._build_body(event)
        )

    def _format_value(self, value: Any) -> str:
        """Convert an event field into a compact, readable email value."""

        # Avoid embedding image/frame data in email notifications.
        if isinstance(value, Frame):
            return "<Frame omitted>"

        # Avoid dumping image arrays, embeddings, or other large numeric data.
        if isinstance(value, np.ndarray):
            return (
                f"<ndarray shape={value.shape}, "
                f"dtype={value.dtype}, omitted>"
            )

        if isinstance(value, np.generic):
            return str(value.item())

        if value is None:
            return "None"

        if isinstance(value, Enum):
            return str(value.value)

        if isinstance(value, (datetime, date, UUID)):
            return str(value)

        # Handle Pydantic models.
        if hasattr(value, "model_dump") and callable(value.model_dump):
            return self._format_value(value.model_dump(mode="json"))

        # Handle dataclasses without converting large nested values blindly.
        if is_dataclass(value) and not isinstance(value, type):
            items = (
                f"{item.name}={self._format_value(getattr(value, item.name))}"
                for item in fields(value)
            )
            return "{" + ", ".join(items) + "}"

        if isinstance(value, dict):
            items = (
                f"{key}: {self._format_value(item)}"
                for key, item in value.items()
            )
            return "{" + ", ".join(items) + "}"

        if isinstance(value, (list, tuple, set)):
            return "[" + ", ".join(
                self._format_value(item) for item in value
            ) + "]"

        # Avoid accidentally including multi-line output in an email field.
        return str(value).replace("\r", "\\r").replace("\n", "\\n")


    def _build_body(self, event: BaseEvent) -> str:
        lines = [
            f"Event:     {type(event).__name__}",
            f"Camera:    {event.camera_id}",
            f"Frame:     {event.frame_id}",
            f"Timestamp: {event.timestamp}",
        ]

        excluded_fields = {"camera_id", "frame_id", "timestamp"}

        for field, value in event.__dict__.items():
            if field in excluded_fields or field.startswith("_"):
                continue

            lines.append(f"{field}: {self._format_value(value)}")

        return "\n".join(lines)

    def _send(self, message: MailMessage) -> None:
        msg = MIMEText(message.body)
        msg["Subject"] = message.subject
        msg["From"]    = message.sender
        msg["To"]      = ", ".join(message.to)

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            if self.password:
                server.login(message.sender, self.password)
            server.sendmail(
                message.sender,
                message.to,
                msg.as_string()
            )

if __name__ == '__main__':
    from trackey.core.events.types import CountExceededEvent
    from datetime import datetime
    password = os.getenv("TRACKEY_SMTP_PASSWORD")
    sub = MailSubscriber(["m.emad4798@gmail.com", "m.emad7798@gmail.com"],
                         "m.emad7798@gmail.com",
                         password=password)
    sub.on_event(CountExceededEvent(frame_id=200,
                                    camera_id='1',
                                    timestamp=datetime.now(),
                                    count=20,
                                    zone_name="Entrance"))
