from abc import ABC, abstractmethod
from trackey.data.schemas.frame import Frame
from trackey.data.schemas.track import Track
from trackey.data.schemas.view import GlobalStateBox, GlobalStateBoxPlacement
from uuid import UUID
from typing import Union, Optional


class OutputViewer(ABC):
    @abstractmethod
    def show(self, frame: Optional[Frame], tracks: list[Track]):
        pass

    @abstractmethod
    def add_global_state_box(self, placement: GlobalStateBoxPlacement) -> UUID:
        pass

    @abstractmethod
    def add_global_state(self, global_state_box: UUID, state_name: str, value: Union[int, float, str]):
        pass

    @abstractmethod
    def open(self) -> bool:
        pass

    @abstractmethod
    def close(self):
        pass
