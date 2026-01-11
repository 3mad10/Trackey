from abc import ABC, abstractmethod
import numpy as np
import cv2
from trackey.data.schemas.detection import Detection
from trackey.data.schemas.frame import Frame


class Drawable(ABC):
    @abstractmethod
    def draw(self, image: np.ndarray) -> None:
        pass


class BBoxDrawable(Drawable):
    def __init__(self, bbox, color=(0, 255, 0), thickness=2):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.color = color
        self.thickness = thickness

    def draw(self, image):
        x1, y1, x2, y2 = self.bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), self.color, self.thickness)


class PointDrawable(Drawable):
    def __init__(self, point, radius=3, color=(0, 0, 255)):
        self.point = point  # (x, y)
        self.radius = radius
        self.color = color

    def draw(self, image):
        cv2.circle(image, self.point, self.radius, self.color, -1)


class KeypointsDrawable(Drawable):
    def __init__(self, keypoints, color=(255, 0, 0)):
        self.keypoints = keypoints  # list[(x, y)]
        self.color = color

    def draw(self, image):
        for pt in self.keypoints:
            cv2.circle(image, pt, 2, self.color, -1)


def detection_to_drawables(det: Detection, frame: Frame):
    drawables = []

    if det.bbox:
        bbox = det.bbox.to_pixel_xyxy(frame.width, frame.height)
        drawables.append(BBoxDrawable(bbox))

    if det.points:
        drawables.append(PointDrawable(det.points))

    if det.keypoints:
        drawables.append(KeypointsDrawable(det.keypoints))

    return drawables

