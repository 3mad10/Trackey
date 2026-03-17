from abc import ABC, abstractmethod
import numpy as np
import cv2
from typing import List, Tuple
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


class PolygonDrawable(Drawable):
    """Draw polygon/zone on frame"""
    
    def __init__(self, 
                 points: List[Tuple[float, float]],
                 frame_width: int,
                 frame_height: int,
                 color: Tuple[int, int, int] = (255, 0, 0),
                 thickness: int = 2,
                 filled: bool = False,
                 alpha: float = 0.3,
                 label: str = None):
        """
        Args:
            points: List of (x, y) in normalized coords (0-1)
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            color: BGR color tuple
            thickness: Line thickness (ignored if filled=True)
            filled: If True, draw filled polygon with transparency
            alpha: Transparency (0=invisible, 1=opaque) for filled polygons
            label: Optional text label for the zone
        """
        self.points = points
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.color = color
        self.thickness = thickness
        self.filled = filled
        self.alpha = alpha
        self.label = label
    
    def draw(self, image: np.ndarray):
        """Draw polygon on image"""
        # Convert normalized points to pixel coordinates
        pixel_points = []
        for x, y in self.points:
            px = int(x * self.frame_width)
            py = int(y * self.frame_height)
            pixel_points.append((px, py))
        
        pts = np.array(pixel_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        if self.filled:
            # Draw filled polygon with transparency
            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], self.color)
            cv2.addWeighted(overlay, self.alpha, image, 1 - self.alpha, 0, image)
        else:
            # Draw polygon outline
            cv2.polylines(image, [pts], isClosed=True, color=self.color, 
                         thickness=self.thickness)
        
        # Draw label if provided
        if self.label and len(pixel_points) > 0:
            # Position label at top-left corner of polygon
            label_x, label_y = pixel_points[0]
            cv2.putText(
                image,
                self.label,
                (label_x, label_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.color,
                2,
                cv2.LINE_AA
            )


class LineDrawable(Drawable):
    """Draw line on frame"""
    
    def __init__(self,
                 start: Tuple[float, float],  # Normalized
                 end: Tuple[float, float],    # Normalized
                 frame_width: int,
                 frame_height: int,
                 color: Tuple[int, int, int] = (255, 0, 0),  # Blue
                 thickness: int = 3,
                 label: str = None):
        """
        Args:
            start: Start point (x, y) normalized
            end: End point (x, y) normalized
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            color: BGR color
            thickness: Line thickness
            label: Optional label
        """
        self.start = start
        self.end = end
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.color = color
        self.thickness = thickness
        self.label = label
    
    def draw(self, image: np.ndarray):
        """Draw line on image"""
        # Convert to pixels
        start_px = (int(self.start[0] * self.frame_width),
                   int(self.start[1] * self.frame_height))
        end_px = (int(self.end[0] * self.frame_width),
                 int(self.end[1] * self.frame_height))
        
        # Draw line
        cv2.line(image, start_px, end_px, self.color, self.thickness)
        
        # Draw label
        if self.label:
            mid_x = (start_px[0] + end_px[0]) // 2
            mid_y = (start_px[1] + end_px[1]) // 2
            cv2.putText(
                image,
                self.label,
                (mid_x, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.color,
                2,
                cv2.LINE_AA
            )


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
        keypoints = det.keypoints.to_pixel_xy(frame.width, frame.height)
        bbox = det.keypoints.as_bbox().to_pixel_xyxy(frame.width, frame.height)
        drawables.append(KeypointsDrawable(keypoints))
        drawables.append(BBoxDrawable(bbox))

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

