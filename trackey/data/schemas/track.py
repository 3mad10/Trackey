from pydantic import BaseModel, Field
from datetime import date, datetime
from uuid import UUID
from typing import TYPE_CHECKING, Annotated, List
from .detection import Detection
from enum import Enum


class Track(BaseModel):
    id: UUID
    detections: List[Detection]
    confidence: str = Field(ge=0, le=1, description="Confidence of the detection")
    last_seen: datetime



