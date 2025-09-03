from trackey.data.schemas.track import Track
from trackey.data.schemas.detection import Detection
from trackey.core.base.TrackerBase import TrackerBase
from trackey.data.schemas.frame import Frame
from typing import Optional
from typing import List
from datetime import datetime, timezone


class DeepSortTracker(TrackerBase):
    def __init__(self, **kwargs):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ModuleNotFoundError:
            print("Run \'pip install ultralytics\' to run yolo detector")
        self.tracks = []
        self.tracker = DeepSort(**kwargs)

    def update(self, detections: List[Detection],
               frame: Optional[Frame] = None) -> List[Track]:
        if frame is None:
            raise Exception(
                "The frame is needed as input for DeepSort tracker"
                )
        deepsort_detections = [
            (list(d.bbox.to_pixel_xyxy(frame.width, frame.height)),
             d.confidence, d.class_id) for d in detections]
        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
        for track in tracks:
            conf = 1 if track.is_confirmed else 0
            track_id = track.track_id
            existing_track = self._get_existing_id(track_id)
            if existing_track:
                existing_track.detections += detections
                existing_track.last_seen = datetime.now(timezone.utc)
                existing_track.confidence = conf
            else:
                self.tracks.append(Track(confidence=conf,
                                     detections=detections,
                                     private_id=track.track_id))

    def get_tracks(self) -> List[Track]:
        return self.tracks

    def _get_existing_id(self, track_id):
        #TODO Optimize search by having a tree or sorted ids
        if len(self.tracks) > 0:
            for track in self.tracks:
                if track.private_id == track_id:
                    return track
        else:
            return None


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
