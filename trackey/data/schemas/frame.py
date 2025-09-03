from pydantic import BaseModel, field_validator
import numpy as np


class Frame(BaseModel):
    frame: np.ndarray
    width: int
    height: int

    model_config = {
        "arbitrary_types_allowed": True
    }

    def __getattr__(self, item):
        """Delegate missing attributes (like .shape) to the numpy array."""
        return getattr(self.frame, item)
    
    def __getitem__(self, key):
        """Allow slicing like frame[y1:y2, x1:x2]."""
        return self.frame[key]

    @field_validator("frame")
    def check_numpy(cls, v):
        if not isinstance(v, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")
        return v
