import cv2
from trackey.core.io.input.base import InputSource
from trackey.data.schemas.frame import Frame


class VideoFileSource(InputSource):
    """
    This class implements the interface InputSource for a video input
    """
    def __init__(self, src: str, **kwargs):
        """
        Parameters
        ----------
        src : str
            path to the input video file
        width : Optional[int]
            width of the output frame, if not passed use opencv default
        height : Optional[int]
            height of the output frame, if not passed use opencv default

        Returns
        -------
        None

        Raises
        ------
        None
        """
        super().__init__(**kwargs)
        self.path = src
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.path)
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.is_open = True
        return self.is_open

    def read(self) -> Frame:
        """
        Read the current frame

        Parameters
        ----------
        None

        Returns
        -------
        frame
            A frame object (np.ndarray, width, height)

        Raises
        ------
        RuntimeError
            Failed to open video
        """
        if not self.is_open:
            raise RuntimeError("Video is not opened")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from video file")
        # Wrap in your Frame schema
        h, w = frame.shape[:2]
        return Frame(frame=frame, width=w, height=h)

    def release(self):
        if self.cap:
            self.cap.release()



    # def _getStream(self, srcUrl: str):
    #     import yt_dlp
    #     # Extract the direct video stream URL
    #     ydl_opts = {'format': 'best'}
    #     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    #         info = ydl.extract_info(srcUrl, download=False)
    #         stream_url = info['url']
    #     return stream_url

    # def _is_url(self, src: str):
    #     return re.match(r'^(?:http|ftp)s?://', src) is not None


if __name__ == "__main__":
    inputSrc = VideoFileSource("https://www.youtube.com/watch?v=ysyYPf-pYZ8")
    while True:
        frame = inputSrc.readFrame()

        # Display the frame (optional)
        cv2.imshow('Video Frame', frame.frame)

        # Press 'q' to exit the video display
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    inputSrc.close()
