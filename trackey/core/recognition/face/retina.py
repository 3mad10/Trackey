import logging
from typing import List, Optional

import numpy as np
import onnxruntime as ort

from trackey.core.interfaces.detector import Detector
from trackey.core.register import register_detector
from trackey.data.schemas.detection import Detection, BoundingBox
from trackey.data.schemas.frame import Frame
from trackey.core.pipeline.constants import UNKOWN_DETECTION_ID

logger = logging.getLogger(__name__)


@register_detector("retinaface")
class RetinaFaceDetector(Detector):
    """
    Whole-frame face detector using InsightFace's RetinaFace model.

    Scans the entire frame once per call. Use this when you need
    to catch faces that may not have an associated body track
    (e.g. partial occlusion, body detector miss) and pair with
    FaceAssociationNode downstream.

    For most low-to-moderate density deployments, prefer
    RetinaFaceCroppedDetector which runs per-track and needs no
    association step.
    """
    #TODO: Add an tutomated way to use cude/check deps for cuda run
    def __init__(self,
                 model_name:      str = "buffalo_s",
                 device:          str = "cuda",
                 det_size:        int = 640,
                 conf_threshold:  float = 0.5,
                 class_id:        int = UNKOWN_DETECTION_ID):
        """
        Args:
            model_name:     InsightFace model pack. "buffalo_l" is the
                             standard general-purpose pack (detection +
                             recognition + landmarks). "buffalo_s" is
                             a smaller/faster variant for edge devices.
            device:         "cpu" or "cuda".
            det_size:       Detector input resolution (square). Larger
                             = more accurate on small/distant faces,
                             slower. 640 is a good default.
            conf_threshold: Minimum detection confidence to keep.
        """
        try:
            from insightface.app import FaceAnalysis
        except ModuleNotFoundError as e:
            logger.error(
                "[RetinaFaceDetector] Run "
                "'pip install insightface onnxruntime-gpu' "
                "to use the RetinaFace detector"
            )
            raise e

        self.conf_threshold = conf_threshold
        self.class_id = class_id
        ctx_id = 0 if device == "cuda" else -1
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    def detect(self, frame: Frame) -> List[Detection]:
        """
        Detect faces in the full frame.

        Returns Detection objects with normalized bbox coordinates,
        class_name="face", and 5-point landmarks stored in metadata
        (useful for face alignment before recognition models).
        """
        faces = self.app.get(frame.frame)

        h, w = frame.height, frame.width
        detections: List[Detection] = []

        for face in faces:
            if face.det_score < self.conf_threshold:
                continue

            x1, y1, x2, y2 = face.bbox
            bbox = BoundingBox(
                cx=float((x1 + x2) / 2) / w,
                cy=float((y1 + y2) / 2) / h,
                w=float(x2 - x1) / w,
                h=float(y2 - y1) / h,
            )

            landmarks = None
            if face.kps is not None:
                landmarks = [
                    (float(x) / w, float(y) / h)
                    for x, y in face.kps
                ]
                
            detections.append(Detection(
                bbox=bbox,
                confidence=float(face.det_score),
                class_id=self.class_id,
                class_name="face",
                metadata={"landmarks": landmarks} if landmarks else None,
            ))

        return detections

    def close(self) -> None:
        pass


if __name__=='__main__':
    import cv2
    image_path = "/home/mohamed-emad/Pictures/ME,MYSELF&I/IMG_20260411_135344.jpg"  # Replace with the actual path to your image
    image = cv2.imread(image_path)
    detector = RetinaFaceDetector()
    frame = Frame(frame = image)
    print(detector.detect(frame=frame))