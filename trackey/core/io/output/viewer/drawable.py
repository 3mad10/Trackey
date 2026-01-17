from abc import ABC, abstractmethod
import numpy as np
import cv2
from trackey.data.schemas.detection import Detection
from trackey.data.schemas.frame import Frame
from trackey.data.schemas.track import Track


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


class TextDrawable(Drawable):
    def __init__(self, text: str, position: tuple[int, int], color=(255, 255, 255), thickness=1, font_scale=0.5):
        self.text = text
        self.position = position
        self.color = color
        self.thickness = thickness
        self.font_scale = font_scale

    def draw(self, image):
        cv2.putText(
            image,
            self.text,
            self.position,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            self.color,
            self.thickness,
            cv2.LINE_AA,
        )


def get_metadata_drawables(metadata: dict, bbox_xyxy: tuple[int, int, int, int]) -> list[Drawable]:
    drawables = []
    if not metadata:
        return drawables

    x1, y1, _, _ = bbox_xyxy

    # Start drawing text above the bbox
    text_y = y1 - 10

    for key, value in metadata.items():
        text = f"{key}: {value}"
        drawables.append(TextDrawable(text, (x1, text_y)))
        text_y -= 15  # Move up for the next line

    return drawables


def detection_to_drawables(det: Detection, frame: Frame):
    drawables = []

    if det.bbox:
        bbox = det.bbox.to_pixel_xyxy(frame.width, frame.height)
        drawables.append(BBoxDrawable(bbox))

        if det.class_name:
            drawables.extend(get_metadata_drawables({"Class: ": det.class_name}, bbox))
        if det.metadata:
            drawables.extend(get_metadata_drawables(det.metadata, bbox))

    if det.points:
        drawables.append(PointDrawable(det.points))

    if det.keypoints:
        drawables.append(KeypointsDrawable(det.keypoints))

    return drawables


def track_to_drawables(track: Track, frame: Frame):
    if not track.detections:
        return []

    det = track.detections[-1]
    # Use existing detection logic
    drawables = detection_to_drawables(det, frame)
    
    # Add track-specific metadata if available
    # We need the bbox to position the text, so we check if we have one from the detection
    if det.bbox and track.metadata:
        bbox = det.bbox.to_pixel_xyxy(frame.width, frame.height)
        # Shift track metadata further up if detection metadata exists
        # This is a simple implementation, might overlap if detection emits a lot of text
        # But for now it's okay.
        # Ideally we pass current text_y or similar.
        
        # Let's count existing TextDrawables to guess offset? 
        # Or just draw above.
        
        # A simpler way: merge metadata? No, they might collide keys.
        # Let's just generate drawables and append.
        
        # Heuristic: detection metadata usually draws immediately above bbox.
        # Let's draw track metadata even higher?
        # Or let's make get_metadata_drawables take a start_y offset?
        
        # For this iteration, let's just use get_metadata_drawables.
        # Note: if both exist, they will overlap unless we adjust.
        # Let's perform a small adjustment.
        
        offset_y = 0
        if det.metadata:
            offset_y = len(det.metadata) * 15
            
        x1, y1, _, _ = bbox
        initial_text_y = y1 - 10 - offset_y
        
        for key, value in track.metadata.items():
            text = f"{key}: {value}"
            drawables.append(TextDrawable(text, (x1, initial_text_y)))
            initial_text_y -= 15

    return drawables

