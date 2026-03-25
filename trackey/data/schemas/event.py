from pydantic import BaseModel, Field
from datetime import datetime, timezone


class Event(BaseModel):
    subject: str = Field(description="Subject of the event.")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
