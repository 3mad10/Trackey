import numpy as np
from pydantic import BaseModel, field_validator
from trackey.data.schemas.geometry import Zone
from typing import List

class Frame(BaseModel):
    frame: np.ndarray

    model_config = {
        "arbitrary_types_allowed": True
    }

    @property
    def height(self) -> int:
        return self.frame.shape[0]

    @property
    def width(self) -> int:
        return self.frame.shape[1]

    def __getattr__(self, item):
        return getattr(self.frame, item)

    def __getitem__(self, key):
        return self.frame[key]

    @field_validator("frame")
    def check_numpy(cls, v):
        if not isinstance(v, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")
        return v


