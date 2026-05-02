import smtplib
import numpy as np

from typing import List, Optional

from trackey.core.interfaces.subscriber import Subscriber
from trackey.core.events.mails.message import MailMessage
from trackey.core.events.types import CountExceededEvent
from trackey.data.schemas.event import BaseEvent



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
        message = self._format(event)
        self._send(message)

    def _format(self, event: BaseEvent) -> MailMessage:
        return MailMessage(
            sender=self.sender,
            to=self.to,
            subject=f"[Trackey] {type(event).__name__}",
            body=self._build_body(event)
        )

    def _build_body(self, event: BaseEvent) -> str:
        lines = [
            f"Camera:    {event.camera_id}",
            f"Frame:     {event.frame_id}",
            f"Timestamp: {event.timestamp}",
        ]
        for field, value in event.__dict__.items():
            if field not in ("camera_id", "frame_id", "timestamp"):
                if not isinstance(value, np.ndarray):
                    lines.append(f"{field}: {value}")
        return "\n".join(lines)

    def _send(self, message: MailMessage) -> None:
        import smtplib
        from email.mime.text import MIMEText

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
    sub = MailSubscriber("mail2")
    sub.on_event(CountExceededEvent(subject="Count Limit", count=20, threshold=19))
