import cv2
import time
from urllib.parse import urlparse
from trackey.core.io.input.base import InputSource
from trackey.data.schemas.frame import Frame


class VideoFileSource(InputSource):
    def __init__(self, url: str, **kwargs):
        super().__init__(**kwargs)
        self.url = url
        self.cap = None

    def _validate_url(self):
        parsed = urlparse(self.url)

        if parsed.scheme != "rtsp":
            raise ValueError("RTSP URL must start with rtsp://")

        if not parsed.hostname:
            raise ValueError("RTSP URL missing hostname")

        if parsed.port is None:
            # not fatal, but good to warn or default
            pass

    def open(self) -> bool:
        if self.is_open:
            return True

        self.cap = cv2.VideoCapture(
            self.url,
            cv2.CAP_FFMPEG
        )

        start = time.time()
        timeout = 5.0

        while not self.cap.isOpened():
            if time.time() - start > timeout:
                self.release()
                return False
            time.sleep(0.1)

        # Try reading a few frames to ensure stream is valid
        for _ in range(5):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.is_open = True
                return True

        # Opened but no frames → invalid stream
        self.release()
        return False

    def read(self) -> Frame:
        if not self.is_open:
            return None

        ret, frame = self.cap.read()

        if not ret or frame is None:
            # stream died or frame decode failed
            self.is_open = False
            return None
        h, w = frame.shape[:2]
        return Frame(frame, width=w, height=h)

    def release(self):
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    pass
