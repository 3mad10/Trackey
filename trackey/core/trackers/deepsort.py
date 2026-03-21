import logging
from typing import Optional, List, Tuple
from datetime import datetime, timezone
from collections import deque

from trackey.core.interfaces.tracker import Tracker
from trackey.data.schemas.track import Track
from trackey.data.schemas.detection import Detection
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
        self.tracks: dict[int, Track] = {}

    def update(self, detections: List[Detection], frame: Optional[Frame]) -> List[Track]:
        if frame is None:
            raise Exception("Frame required")

        ds_inputs = [
            (list(d.bbox.to_pixel_xywh(frame.width, frame.height)),
            d.confidence, d.class_id)
            for d in detections if d.bbox
        ]

        ds_tracks = self.tracker.update_tracks(ds_inputs, frame=frame.frame)

        alive_ids = set()
        now = datetime.now(timezone.utc)

        for ds_track in ds_tracks:
            if not ds_track.is_confirmed():
                continue

            track_id = ds_track.track_id
            alive_ids.add(track_id)

            det = self._attach_detection(
                ds_track.to_ltrb(),
                detections,
                frame,
            )

            if det is None:
                continue

            if track_id in self.tracks:
                t = self.tracks[track_id]
                t.detections.append(det)
                t.last_seen = now
            else:
                self.tracks[track_id] = Track(
                    tracker_id=track_id,
                    detections=deque([det], maxlen=30),
                    confidence=1.0,
                )

        # REMOVE DEAD TRACKS
        for tid in list(self.tracks.keys()):
            if tid not in alive_ids:
                del self.tracks[tid]

        return list(self.tracks.values())


    def get_tracks(self) -> List[Track]:
        return list(self.tracks.values())

    def _get_existing_id(self, track_id):
        if track_id in self.tracks:
            return self.tracks[track_id]
        return None

    def _attach_detection(
            self,
            track_ltrb,
            detections: list[Detection],
            frame: Frame,
            iou_thresh=0.7,
            ) -> Detection | None:

        best_det = None
        best_iou = 0.0

        for det in detections:
            if det.bbox is None:
                continue

            det_ltrb = det.bbox.to_pixel_xyxy(frame.width, frame.height)
            iou = self._iou_ltrb(track_ltrb, det_ltrb)

            if iou > best_iou:
                best_iou = iou
                best_det = det

        if best_iou < iou_thresh:
            return None

        return best_det

    def _iou_ltrb(
            self,
            box_a: Tuple[float, float, float, float],
            box_b: Tuple[float, float, float, float],
            ) -> float:
        """
        Compute Intersection-over-Union (IoU) between two LTRB boxes.

        Parameters
        ----------
        box_a : (l, t, r, b)
        box_b : (l, t, r, b)

        Returns
        -------
        float
            IoU value in [0, 1]
        """

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        # Intersection box
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter_area = iw * ih

        # Areas
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union_area = area_a + area_b - inter_area

        if union_area <= 0.0:
            return 0.0

        return inter_area / union_area


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
