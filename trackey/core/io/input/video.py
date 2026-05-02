import cv2
import time
from typing import Optional
from trackey.core.interfaces.source import InputSource
from trackey.data.schemas.frame import Frame


class VideoFileSource(InputSource):
    def __init__(self,
                 path: str,
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 target_fps: Optional[float] = None,
                 drop_strategy: str = "block"):
        self.path          = path
        self.width         = width
        self.height        = height
        self.target_fps    = target_fps
        self.drop_strategy = drop_strategy
        self.cap           = None
        self.is_open       = False
        self._source_fps   = None
        self._frame_delay  = None
        self._last_read    = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"[VideoFileSource] Failed to open: {self.path}"
            )
        self._source_fps  = self.cap.get(cv2.CAP_PROP_FPS)
        fps               = self.target_fps or self._source_fps
        self._frame_delay = 1.0 / fps if fps > 0 else None
        self._last_read   = time.monotonic()
        self.is_open      = True
        return True

    def read(self) -> Optional[Frame]:
        if not self.is_open:
            raise RuntimeError("[VideoFileSource] Source not opened")

        if self._frame_delay:
            self._wait()

        ret, frame = self.cap.read()
        if not ret:
            return None  # end of video — not an error

        self._last_read = time.monotonic()

        # resize if requested
        if self.width or self.height:
            frame = self._resize(frame)

        h, w = frame.shape[:2]
        return Frame(frame=frame, width=w, height=h)

    def release(self) -> None:
        if self.cap:
            self.cap.release()
        self.is_open = False

    @property
    def camera_id(self) -> str:
        return self.path

    @property
    def source_fps(self) -> Optional[float]:
        return self._source_fps

    def _wait(self) -> None:
        elapsed   = time.monotonic() - self._last_read
        remaining = self._frame_delay - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _resize(self, frame) -> any:
        h, w = frame.shape[:2]
        if self.width and self.height:
            return cv2.resize(frame, (self.width, self.height))
        if self.width:
            ratio  = self.width / w
            return cv2.resize(frame, (self.width, int(h * ratio)))
        ratio = self.height / h
        return cv2.resize(frame, (int(w * ratio), self.height))