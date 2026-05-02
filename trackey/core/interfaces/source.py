from abc import ABC, abstractmethod
from typing import Tuple

from trackey.data.schemas.frame import Frame


class InputSource(ABC):
    def __init__(self, **kwargs):
        self.is_open = False
        self.width = None
        self.height = None

        if "width" in kwargs:
            self.width = kwargs["width"]
        if "height" in kwargs:
            self.height = kwargs["height"]

    @abstractmethod
    def open(self) -> bool:
        pass

    @abstractmethod
    def read(self) -> Frame:
        pass

    @abstractmethod
    def release(self):
        pass

    @property
    @abstractmethod
    def camera_id(self) -> str:
        pass
