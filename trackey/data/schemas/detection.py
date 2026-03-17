from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone
from uuid import UUID, uuid4
from typing import List, Optional, Tuple
import numpy as np
from pydantic import model_validator, field_validator


class Point(BaseModel):
    """Single normalized point (0-1 range)"""
    x: float = Field(description="X coordinate (normalized)")
    y: float = Field(description="Y coordinate (normalized)")


    @field_validator("x", "y", mode="before")
    @classmethod
    def clamp_input(cls, v):
        return max(0.0, min(1.0, float(v)))

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


class Keypoint(BaseModel):
    name: str = Field(description="Keypoint semantic name (e.g. left_eye)")
    point: Point
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

    def to_pixel(self, w: int, h: int):
        return self.point.to_pixel(w, h)


class Keypoints(BaseModel):
    items: List[Keypoint]
    format: str = Field(
        default="coco",
        description="Keypoint format convention"
    )

    def as_numpy(self) -> np.ndarray:
        """(N, 3) -> x, y, conf"""
        return np.array([
            [kp.point.x, kp.point.y, kp.confidence]
            for kp in self.items
        ])

    def as_bbox(self) -> BoundingBox:
        valid_points = [
            kp.point for kp in self.items
            if 0 <= kp.point.x <= 1 and 0 <= kp.point.y <= 1
        ]

        if not valid_points:
            raise ValueError("No valid keypoints to compute bbox")

        xs = [p.x for p in valid_points]
        ys = [p.y for p in valid_points]

        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        width = max_x - min_x
        height = max_y - min_y

        # expand upward to include head
        min_y = max(0.0, min_y - 0.5 * height)

        # expand downward a bit for legs
        max_y = min(1.0, max_y + 0.15 * height)

        height = max_y - min_y

        cx = min_x + width / 2
        cy = min_y + height / 2

        return BoundingBox(
            cx=cx,
            cy=cy,
            w=min(width, 1.0),
            h=min(height, 1.0)
        )
    
    def to_pixel_xy(self, img_w: int, img_h: int) -> List[Keypoint]:
        keypoints = []
        for item in self.items:
            keypoints.append((item.point.to_pixel(img_width=img_w, img_height=img_h)))
        return keypoints

class Detection(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    class_id: int = Field(ge=0)
    class_name: str = Field(max_length=512, description="Class label name")
    confidence: float = Field(ge=0, le=1)

    bbox: Optional[BoundingBox] = None #TODO: Make bounding box mandatory
    points: Optional[List[Point]] = None
    keypoints: Optional[Keypoints] = None

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    features: Optional[List[float]] = None
    metadata: Optional[dict] = None

    @model_validator(mode="after")
    def check_geometry(self):
        geometries = [
            self.bbox is not None,
            bool(self.points),
            self.keypoints is not None,
        ]
        if sum(geometries) != 1:
            raise ValueError("Detection must have exactly one geometry type")
        return self
