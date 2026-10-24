import cv2
import time
from urllib.parse import urlparse
from trackey.core.interfaces.source import InputSource
from trackey.data.schemas.frame import Frame
import logging

logger = logging.getLogger(__name__)

class RtspSource(InputSource):
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
            self.release()

            # auto-reconnect backoff
            logger.warning(f"[RtspSource] Connection lost to {self.url}. Attempting to reconnect...")
            backoff = 1.0
            for attempt in range(1, 6):
                time.sleep(backoff)
                if self.open():
                    logger.info(f"[RtspSource] Reconnected successfully to {self.url}")
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        break
                backoff = min(backoff * 2, 10.0)
                logger.warning(f"[RtspSource] Reconnect attempt {attempt} failed.")

            if not self.is_open:
                return None

        if self.width and self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        return Frame(frame=frame)

    def release(self):
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    pass
