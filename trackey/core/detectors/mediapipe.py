from typing import List
from pathlib import Path
import platform
import urllib.request

from trackey.core.interfaces.detector import Detector
from trackey.data.schemas.detection import Detection, BoundingBox
from trackey.data.schemas.frame import Frame
from trackey.core.register import register_detector

# TODO: Add to config
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


@register_detector('mediapipe')
class MPLandmarkDetector(Detector):
    def __init__(self, weights="pose_landmarker_lite.task"):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ModuleNotFoundError:
            print("Run \'pip install mediapipe\' to run mediapipe landmark detector")

        # TODO: Add to statics
        MODEL_PATH = f"trackey/core/models/{weights}"
        weights = self._download_model(MODEL_PATH)
        self.mp = mp
        BaseOptions = self.mp.tasks.BaseOptions
        PoseLandmarker = self.mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = self.mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = self.mp.tasks.vision.RunningMode
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=weights),
            running_mode=VisionRunningMode.IMAGE)
        self.model = PoseLandmarker.create_from_options(options)
        self.names = None

    def detect(self, frame: Frame) -> List[Detection]:
        """
        Detect the objects in one frame and return it as a list of Detection object

        Parameters:
        Frame (numpy ndarray, w, h): current frame.

        Returns:
        List[Detection]: List of Detection objects.
        """
        results = self.model.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame.frame))
        return results

    def close(self):
        if self.model:
            self.model.close()

    def _download_model(self, model_path: str) -> str:
        if Path(model_path).exists():
            return model_path

        urllib.request.urlretrieve(MODEL_URL, model_path)

        return model_path


def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Draw MediaPipe pose landmarks directly on an image using OpenCV.
    
    Parameters:
    - rgb_image: numpy.ndarray, RGB image
    - detection_result: result from MPLandmarkDetector.detect()
    
    Returns:
    - annotated_image: numpy.ndarray with landmarks drawn
    """
    annotated_image = np.copy(rgb_image)
    pose_landmarks_list = detection_result.pose_landmarks

    # Define connections for the pose (same as MediaPipe)
    POSE_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
        (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
        (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(11,23),
        (12,24),(23,24),(23,25),(24,26),(25,27),(26,28),(27,29),(28,30),(29,31),(30,32)
    ]

    for landmarks in pose_landmarks_list:
        # Convert normalized coordinates to pixel coordinates
        h, w, _ = annotated_image.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # Draw joints
        for x, y in points:
            cv2.circle(annotated_image, (x, y), 3, (0, 255, 0), -1)  # green dots

        # Draw connections
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(annotated_image, points[start_idx], points[end_idx], (0, 0, 255), 2)  # red lines

    return annotated_image


if __name__=='__main__':
    import cv2
    import numpy as np

    # Load image
    image_path = "C:/Users/Mohamed Emad/OneDrive/Pictures/New York/20221203_203840.jpg"
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # MediaPipe expects RGB

    # Initialize detector
    detector = MPLandmarkDetector()
    frame = Frame(frame=rgb_image, width=rgb_image.shape[1], height=rgb_image.shape[0])

    # Detect landmarks
    results = detector.detect(frame)

    # Draw landmarks
    annotated_image = draw_landmarks_on_image(rgb_image, results)

    # Convert back to BGR for OpenCV display
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    display_image = cv2.resize(annotated_image_bgr, (512, 512))
    # Show image
    cv2.imshow('Annotated Image', display_image)
    cv2.waitKey(0)  # Wait indefinitely
    cv2.destroyAllWindows()

    # Close detector to release resources
    detector.close()
