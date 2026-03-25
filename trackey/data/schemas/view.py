from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Literal, Tuple, Optional
from enum import Enum


GlobalStateBoxPlacement = Literal[
    # Top
    "top_left",
    "top_center",
    "top_right",

    # Middle
    "middle_left",
    "middle_center",
    "middle_right",

    # Bottom
    "bottom_left",
    "bottom_center",
    "bottom_right",
]


class GlobalStateBox(BaseModel):
    id: UUID = Field(default_factory= uuid4())
    cx: float = Field(ge=0.0, le=1.0, description="Center X (normalized)")
    cy: float = Field(ge=0.0, le=1.0, description="Center Y (normalized)")
    w: float = Field(gt=0.0, le=1.0, description="Width (normalized)")
    h: float = Field(gt=0.0, le=1.0, description="Height (normalized)")

    @property
    def center(self) -> Tuple[float, float]:
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

    def to_pixel_xyxy(self, img_width: int, img_height: int):
        """Convert to pixel coordinates in xyxy format"""
        x1, y1, x2, y2 = self.xyxy
        return (
            int(x1 * img_width), int(y1 * img_height),
            int(x2 * img_width), int(y2 * img_height)
        )

    def to_pixel_xywh(self, img_width: int, img_height: int):
        """Convert to pixel coordinates in xywh format"""
        x1, y1, w, h = self.xywh
        return (
            int(x1 * img_width), int(y1 * img_height),
            int(w * img_width), int(h * img_height)
        )

    class Config:
        # Enable ORM mode for database integration
        from_attributes = True
        # Custom JSON schema for API documentation
        json_schema_extra = {
            "example": {
                "cx": 0.9,
                "cy": 0.1,
                "w": 0.2,
                "h": 0.2
            }
        }
