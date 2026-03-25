from pydantic import BaseModel, Field
from datetime import datetime, timezone
from uuid import UUID, uuid4
from typing import Tuple


class BoundingBox(BaseModel):
    cx: float = Field(ge=0.0, le=1.0)
    cy: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)

    @property
    def center(self) -> Tuple[float, float]:
        return self.cx, self.cy

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        x1 = self.cx - self.w / 2
        y1 = self.cy - self.h / 2
        x2 = self.cx + self.w / 2
        y2 = self.cy + self.h / 2
        return x1, y1, x2, y2

    @property
    def xywh(self) -> Tuple[float, float, float, float]:
        x1 = self.cx - self.w / 2
        y1 = self.cy - self.h / 2
        return x1, y1, self.w, self.h

    def to_pixel_xyxy(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.xyxy
        return (
            int(x1 * img_w),
            int(y1 * img_h),
            int(x2 * img_w),
            int(y2 * img_h),
        )

    def to_pixel_xywh(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        x1, y1, w, h = self.xywh
        return (
            int(x1 * img_w),
            int(y1 * img_h),
            int(w * img_w),
            int(h * img_h),
        )

class Detection(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    class_id: int = Field(ge=0)
    class_name: str = Field(max_length=512, description="Class label name")
    confidence: float = Field(ge=0, le=1)

    bbox: BoundingBox

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

