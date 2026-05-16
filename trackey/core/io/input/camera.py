import cv2

from trackey.core.interfaces.source import InputSource
from trackey.data.schemas.frame import Frame


class CameraSource(InputSource):
    """
    This class implements the interface InputSource for a webcam input
    """
    def __init__(self, index: int = 0, **kwargs):
        """
        Parameters
        ----------
        index : Optional[int]
            index of the webcam if not passed the default is zero
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
        self.index = index
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.index)
        if self.cap.isOpened():
            self.is_open = True
        else:
            self.is_open = False
        return self.is_open

    def read(self) -> Frame:
        """
        Read the current frame

        Parameters
        ----------
        index : Optional[int]
            index of the webcam if not passed the default is zero
        width : Optional[int]
            width of the output frame, if not passed use opencv default
        height : Optional[int]
            height of the output frame, if not passed use opencv default

        Returns
        -------
        frame
            A frame object (np.ndarray, width, height)

        Raises
        ------
        RuntimeError
            Failed to open webcam
        """
        if not self.is_open:
            raise RuntimeError("Webcam is not open")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from webcam")
        # Wrap in your Frame schema
        h, w = frame.shape[:2]
        return Frame(frame=frame)

    def release(self):
        if self.cap:
            self.cap.release()

    @property
    def camera_id(self):
        return str(self.index)


if __name__ == "__main__":
    inputSrc = CameraSource(index=0, width=1920)
    while True:
        frame = inputSrc.readFrame()

        # Display the frame (optional)
        cv2.imshow('Video Frame', frame.frame)

        # Press 'q' to exit the video display
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    inputSrc.close()
