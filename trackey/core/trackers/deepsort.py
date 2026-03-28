import logging
from typing import Optional, List, Tuple
from datetime import datetime, timezone
from collections import deque

from trackey.core.interfaces.tracker import Tracker
from trackey.data.schemas.track import Track
from trackey.data.schemas.detection import Detection, BoundingBox
from trackey.data.schemas.frame import Frame
from trackey.core.register import register_tracker


logger = logging.getLogger(__name__)

@register_tracker('deepsort')
class DeepSortTracker(Tracker):
    def __init__(self, **kwargs):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ModuleNotFoundError as e:
            logger.error(f"[DeepSortTracker] Run \'pip install deep-sort-realtime\' to run DeepSort tracker")
            raise e
        self.tracker = DeepSort(**kwargs)

    def update(self, detections: List[Detection], frame: Optional[Frame]) -> List[Track]:
        if frame is None:
            raise Exception("[Tracker][DeepSortTracker] Frame required")
        
        # Create list of bbox in original frame width and height
        ds_inputs = []

        for det in detections:

            if det.bbox is None:
                continue
            
            if det.confidence < 0.4:
                continue

            ds_inputs.append((
                list(det.bbox.to_pixel_xywh(
                    frame.width,
                    frame.height
                )),
                det.confidence,
                det.class_name
            ))

        ds_tracks = self.tracker.update_tracks(ds_inputs, frame=frame.frame)

        tracks = []

        now = datetime.now(timezone.utc)

        for ds_track in ds_tracks:

            if not ds_track.is_confirmed():
                continue

            if ds_track.time_since_update > 0:
                continue
            
            l, t, w, h = ds_track.to_ltwh()

            cx = (l + w/2) / frame.width
            cy = (t + h/2) / frame.height
            w = w / frame.width
            h = h / frame.height

            cx = self.clamp(cx)
            cy = self.clamp(cy)
            w = self.clamp(w)
            h = self.clamp(h)

            bbox = BoundingBox(cx=cx, cy=cy, w=w, h=h)
            # print("=============")
            # print("det : ", det)
            tracks.append(
                Track(
                    id=ds_track.track_id,
                    bbox=bbox,
                    confidence=ds_track.get_det_conf() or 1.0,
                    class_name=ds_track.get_det_class(),
                    age=ds_track.age,
                    last_seen=now
                )
            )

        return tracks

    def get_tracks(self) -> List[Track]:
        return list(self.tracks.values())
    
    def clamp(self, v):
        return max(1e-6, min(1.0, float(v)))


if __name__ == '__main__':
    import cv2
    from trackey.core.detectors.yolo import YoloDetector
    image_path = "C:/Users/Mohamed Emad/OneDrive/Pictures/New York/20221203_203840.jpg"  # Replace with the actual path to your image
    image = cv2.imread(image_path)
    detector = YoloDetector()
    tracker = DeepSortTracker()
    detections = detector.detect(image)
    h, w = image.shape[:2]
    # print(detections)
    tracker.update(detections, frame=Frame(frame=image, width=w, height=h))
    print(tracker.get_tracks())
