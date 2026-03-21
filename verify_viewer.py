import cv2
import numpy as np
from uuid import uuid4
from trackey.core.io.output.viewer.opencv_viewer import OpenCVViewer
from trackey.data.schemas.frame import Frame
from trackey.data.schemas.track import Track
from trackey.data.schemas.detection import Detection, BoundingBox

class MockViewer(OpenCVViewer):
    def _render(self, img):
        print("Render called successfully")
        # cv2.imshow would go here

def test_viewer():
    # Setup
    frame_data = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = Frame(frame=frame_data, frame_id=1, timestamp=0.0)
    viewer = MockViewer(window_name="Test")
    
    # Test 1: Detection only
    print("Testing Detection only...")
    bbox = BoundingBox(cx=0.5, cy=0.5, w=0.1, h=0.1)
    det = Detection(
        bbox=bbox, 
        confidence=0.9, 
        class_id=0, 
        class_name="person",
        metadata={"activity": "walking"}
    )
    viewer.show(frame, [det])
    
    # Test 2: Track with metadata
    print("Testing Track with metadata...")
    track = Track(
        tracker_id=uuid4(),
        confidence=0.8,
        metadata={"status": "tracked", "id": "123"}
    )
    track.detections.append(det)
    viewer.show(frame, [track])

    # Test 3: Mixed list
    print("Testing Mixed list...")
    viewer.show(frame, [det, track])
    
    print("All tests passed!")

if __name__ == "__main__":
    test_viewer()
