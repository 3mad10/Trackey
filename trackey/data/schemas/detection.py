from pydantic import BaseModel, Field, validator
from datetime import date, datetime, timezone
from uuid import UUID, uuid4
from typing import List, Optional, Tuple
import numpy as np


class Point(BaseModel):
    """Single normalized point (0-1 range)"""
    x: float = Field(ge=0.0, le=1.0, description="X coordinate (normalized)")
    y: float = Field(ge=0.0, le=1.0, description="Y coordinate (normalized)")

    def to_pixel(self, img_width: int, img_height: int) -> Tuple[int, int]:
        """Convert to pixel coordinates"""
        return (int(self.x * img_width), int(self.y * img_height))

    @classmethod
    def from_pixel(cls, x: int, y: int, img_width: int, img_height: int):
        """Create from pixel coordinates"""
        return cls(x=x/img_width, y=y/img_height)

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y])


class BoundingBox(BaseModel):
    """Normalized bounding box with automatic validation"""
    cx: float = Field(ge=0.0, le=1.0, description="Center X (normalized)")
    cy: float = Field(ge=0.0, le=1.0, description="Center Y (normalized)")
    w: float = Field(gt=0.0, le=1.0, description="Width (normalized)")
    h: float = Field(gt=0.0, le=1.0, description="Height (normalized)")

    @validator('cx', 'cy', 'w', 'h')
    def validate_normalized(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Value must be normalized (0-1), got {v}")
        return v

    @property
    def center(self) -> Tuple[float]:
        return (self.cx, self.cy)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def xyxy(self, img_width: Optional[int] = None,
                img_height: Optional[int] = None) -> Tuple[float, ...]:
        x1 = self.cx - self.w/2
        y1 = self.cy - self.h/2
        x2 = self.cx + self.w/2
        y2 = self.cy + self.h/2

        if img_width and img_height:
            return (int(x1 * img_width), int(y1 * img_height),
                    int(x2 * img_width), int(y2 * img_height))
        return (x1, y1, x2, y2)
    
    @property
    def xywh(self, img_width: Optional[int] = None,
                img_height: Optional[int] = None) -> Tuple[float, ...]:
        x1 = self.cx - self.w/2
        y1 = self.cy - self.h/2
        w = self.w
        h = self.h

        if img_width and img_height:
            return (int(x1 * img_width), int(y1 * img_height),
                    int(w * img_width), int(h * img_height))
        return (x1, y1, w, h)

    class Config:
        # Enable ORM mode for database integration
        from_attributes = True
        # Custom JSON schema for API documentation
        json_schema_extra = {
            "example": {
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.1,
                "h": 0.2
            }
        }


class Detection(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    class_id: int = Field(ge=0)
    class_name: str = Field(max=512, description="Confidence of the detection")
    confidence: float = Field(ge=0, le=1, description="Confidence of the detection")
    bbox: BoundingBox
    points: Optional[List[Point]] = Field(None, description="List of detection points")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # frame_number: int = Field(ge=0, description="Defined Corner Coordinates representation x_min, y_min, x_max, y_max")
    features: Optional[List[float]] = Field(None, description="Feature embedding vector")
    metadata: Optional[dict] = Field(None, description="Additional metadata for detection to be used in post processing")

