from typing import List
from dataclasses import dataclass


@dataclass
class MailMessage:
    sender:  str
    to:      List[str]
    subject: str
    body:    str