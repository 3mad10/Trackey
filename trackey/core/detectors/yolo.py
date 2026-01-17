from typing import List

from trackey.core.interfaces.detector import Detector
from trackey.data.schemas.detection import Detection, BoundingBox
from trackey.data.schemas.frame import Frame
from trackey.core.register import register_detector

@register_detector('yolo')
class YoloDetector(Detector):
    def __init__(self, weights="yolov8n.pt"):
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError:
            print("Run \'pip install ultralytics\' to run yolo detector")
        self.model = YOLO(weights)
        self.weights = weights
        self.names = None

    def detect(self, frame: Frame) -> List[Detection]:
        """
        Detect the objects in one frame and return it as a list of Detection object

        Parameters:
        Frame (numpy ndarray, w, h): current frame.

        Returns:
        List[Detection]: List of Detection objects.
        """
        results = self.model.predict(frame.frame)
        if not self.names:
            self.names = results[0].names
        detections = []
        # We are infering on one frame so get the result of that frame 
        frame_detections = results[0].boxes
        for i, detection in enumerate(frame_detections):
            xywhn = detection.xywhn[0]
            bbox = BoundingBox(
                    cx=float(xywhn[0]),
                    cy=float(xywhn[1]),
                    w=float(xywhn[2]),
                    h=float(xywhn[3]),
                )
            detection = Detection(bbox=bbox,
                                  confidence=detection.conf[0],
                                  class_id=int(detection.cls[0]),
                                  class_name=self.names[int(detection.cls[0])])
            detections.append(detection)
        return detections

    def close(self):
        pass


if __name__=='__main__':
    import cv2
    image_path = "C:/Users/Mohamed Emad/OneDrive/Pictures/New York/20221203_203840.jpg"  # Replace with the actual path to your image
    image = cv2.imread(image_path)
    detector = YoloDetector()
    frame = Frame(frame = image, width=image.shape[1], height=image.shape[0])
    # print(detector.detect(frame=frame))