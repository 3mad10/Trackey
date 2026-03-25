from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pydantic import field_validator

from trackey.data.schemas.detection import BoundingBox


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
            keypoints.append(Keypoint(name=item.name,
                                      point=item.point.to_pixel(img_width=img_w, img_height=img_h),
                                      confidence=item.confidence))
        return keypoints

class Features(BaseModel):
    keypoints: Optional[Keypoints] = None         # pose
    depth: Optional[float] = None                 # depth estimation
    attributes: Dict[str, Any] = Field(default_factory=dict)  # age, gender, color etc.
