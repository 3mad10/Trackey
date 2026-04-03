from pydantic import BaseModel, Field
from datetime import datetime, timezone


class Event(BaseModel):
    subject: str = Field(description="Subject of the event.")
    frame_id: int = Field(description="ID of the Frame the event occured on.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
