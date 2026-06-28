import logging
from typing import Optional, List
from datetime import datetime, timezone
from collections import defaultdict, deque
from dataclasses import replace

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
        self.tracks = defaultdict()

    def update(self, detections: List[Detection], frame: Optional[Frame]) -> List[Track]:
        if frame is None:
            raise Exception("[Tracker][DeepSortTracker] Frame required")

        ds_inputs = self._get_tracker_inputs(detections, frame)

        ds_tracks = self.tracker.update_tracks(ds_inputs, frame=frame.frame)

        tracks = self._wrap_tracks(ds_tracks, frame)

        return tracks
        
    
    def _get_tracker_inputs(self, detections: List[Detection], frame: Frame):
        # Create list of bbox in original frame width and height
        ds_inputs = []

        for det in detections:

            if self._skip_detection(det):
                continue

            ds_inputs.append((
                list(det.bbox.to_pixel_xywh(
                    frame.width,
                    frame.height
                )),
                det.confidence,
                det.class_name
            ))
        return ds_inputs
    
    def _wrap_tracks(self, ds_tracks, frame: Frame):

        tracks = []

        now = datetime.now(timezone.utc)

        for ds_track in ds_tracks:

            if self._skip_track(ds_track):
                continue
            
            bbox = self._get_bbox(ds_track, frame=frame)
            if self._track_exist(ds_track.track_id):
                track: Track = self.tracks[ds_track.track_id]
                track = replace(
                    track,
                    bbox=bbox,
                    history=self._update_history(track.history, bbox),
                    age=ds_track.age,
                    last_seen=now,
                )
                # print("asdsaddasasdasd")
                # print(track)
                tracks.append(track)
            else:
                track: Track = Track(
                        id=ds_track.track_id,
                        bbox=bbox,
                        confidence=ds_track.get_det_conf() or 1.0,
                        class_name=ds_track.get_det_class(),
                        age=ds_track.age,
                        last_seen=now
                    )
                tracks.append(track)
            self.tracks[ds_track.track_id] = track

        return tracks
    
    def _update_history(self, history, bbox):

        new_history = deque(history, maxlen=history.maxlen)

        new_history.append(bbox)

        return new_history
    
    def _track_exist(self, track_id):
        return track_id in self.tracks

    def _skip_detection(self, detection: Detection):
        if detection.bbox is None or detection.confidence < 0.4:
            return True
        else:
            return False

    def _skip_track(self, ds_track):
        if not ds_track.is_confirmed() or ds_track.time_since_update > 0:
            return True
        else:
            return False
    
    def _get_bbox(self, ds_track, frame: Frame):
        l, t, w, h = ds_track.to_ltwh()

        cx = self._clamp((l + w/2) / frame.width)
        cy = self._clamp((t + h/2) / frame.height)
        w = self._clamp(w / frame.width)
        h = self._clamp(h / frame.height)
        return BoundingBox(cx=cx, cy=cy, w=w, h=h)

    def _clamp(self, v):
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
