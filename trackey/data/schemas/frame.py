import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class Frame:
    frame: np.ndarray

    @property
    def height(self) -> int:
        return self.frame.shape[0]

    @property
    def width(self) -> int:
        return self.frame.shape[1]

    @property
    def channels(self) -> int:
        return self.frame.shape[2] if len(self.frame.shape) > 2 else 1

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.width, self.height

