from trackey.core.base.DetectorBase import DetectorBase
from trackey.data.schemas.detection import Detection, BoundingBox
from trackey.data.schemas.frame import Frame
from typing import List


class YoloDetector(DetectorBase):
    def __init__(self, version="yolov8n.pt"):
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError:
            print("Run \'pip install ultralytics\' to run yolo detector")
        self.model = YOLO(version)
        self.version = version
        self.names = None

    def detect(self, frame: Frame) -> List[Detection]:
        """
        Detect the objects in one frame and return it as a list of Detection object

        Parameters:
        Frame (numpy ndarray): current frame.

        Returns:
        List[Detection]: List of Detection objects.
        """
        results = self.model(frame)
        if not self.names:
            self.names = results[0].names
        detections = []
        # We are infering on one frame so get the result of that frame 
        frame_detections = results[0].boxes
        for i, detection in enumerate(frame_detections):
            bbox = BoundingBox(cx=detection.xywhn[0][0] + detection.xywhn[0][2]/2,
                               cy=detection.xywhn[0][1] + detection.xywhn[0][3]/2,
                               w=detection.xywhn[0][2],
                               h=detection.xywhn[0][3])
            detection = Detection(bbox=bbox,
                                  confidence=detection.conf[0],
                                  class_id=int(detection.cls[0]),
                                  class_name=self.names[int(detection.cls[0])])
            detections.append(detection)
        return detections


if __name__=='__main__':
    import cv2
    image_path = "C:/Users/Mohamed Emad/OneDrive/Pictures/New York/20221203_203840.jpg"  # Replace with the actual path to your image
    image = cv2.imread(image_path)
    detector = YoloDetector()
    
    print(detector.detect(image))