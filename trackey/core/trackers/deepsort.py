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
        self.tracker = DeepSort(nn_budget=25,**kwargs)
        self.tracks: dict[int, Track] = {}

    def update(self, detections: List[Detection], frame: Optional[Frame]) -> List[Track]:
        if frame is None:
            raise Exception("[Tracker][DeepSortTracker] Frame required")
        
        now = datetime.now(timezone.utc)
        # Create list of bbox in original frame width and height
        ds_inputs = []
        others = []

        for det in detections:

            if det.bbox is None:
                continue

            ds_inputs.append((
                list(det.bbox.to_pixel_xywh(
                    frame.width,
                    frame.height
                )),
                det.confidence,
                det.class_id
            ))
            others.append(det.class_name)

        ds_tracks = self.tracker.update_tracks(ds_inputs, frame=frame.frame, others=others)

        alive_ids = set()

        for ds_track in ds_tracks:
            if not ds_track.is_confirmed():
                continue

            track_id = ds_track.track_id
            alive_ids.add(track_id)
            
            l, t, bw, bh = ds_track.to_ltwh()

            cx = (l + bw/2) / frame.width
            cy = (t + bh/2) / frame.height
            w = bw / frame.width
            h = bh / frame.height

            cx = self.clamp(cx)
            cy = self.clamp(cy)
            w = self.clamp(w)
            h = self.clamp(h)

            bbox = BoundingBox(cx=cx, cy=cy, w=w, h=h)
            # print("=============")
            # print("det : ", det)
            if track_id not in self.tracks:
                self.tracks[track_id] = Track(
                    tracker_id=track_id,
                    bbox=bbox,
                    confidence=det.confidence if det else 0,
                    last_seen=now,
                    class_name=ds_track.others[0],
                    age=1
                )
            else:
                t = self.tracks[track_id]

                t.bbox = bbox

                if det:
                    t.confidence = det.confidence
                    t.hits += 1
                    t.time_since_update = 0
                else:
                    t.time_since_update += 1

                t.age += 1
                t.last_seen = now
                t.history.append(det)

        # REMOVE DEAD TRACKS
        for tid in list(self.tracks.keys()):
            if tid not in alive_ids:
                del self.tracks[tid]

        return list(self.tracks.values())


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
