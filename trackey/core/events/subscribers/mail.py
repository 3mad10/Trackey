import smtplib

from email.message import EmailMessage
from email.headerregistry import Address

from trackey.core.interfaces.subscriber import Subscriber
from trackey.core.events.types import CountExceededEvent
from trackey.data.schemas.event import Event


class MailSubscriber(Subscriber):

    def __init__(self, subscriber:str):
        self.subscriber = subscriber

    def on_event(self,event:Event):

        if not isinstance(event,CountExceededEvent):
            return

        subject = event.subject

        description = (
            f"Count exceeded: "
            f"{event.count} > {event.threshold}"
        )

        self.send_mail(subject,description)


    def send_mail(self,subject,description):

        msg = EmailMessage()

        msg['Subject'] = subject
        msg['From'] = "m.emad7798@gmail.com"
        msg['To'] = self.subscriber

        msg.set_content(
            f"Trackey Alert\n\n{description}"
        )

        with smtplib.SMTP("smtp.gmail.com",587) as s:

            s.starttls()

            s.login(
                "m.emad7798@gmail.com",
                "mrvz dfdb cpjn ifwx "
            )

            s.send_message(msg)

if __name__ == '__main__':
    sub = MailSubscriber("m.emad4798@gmail.com")
    sub.on_event(CountExceededEvent(subject="Count Limit", count=20, threshold=19))
