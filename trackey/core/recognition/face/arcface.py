import logging

import numpy as np

logger = logging.getLogger(__name__)


class ArcFaceExtractor:
    """Extract face identity embeddings using an ArcFace model."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str = "cuda",
    ):
        try:
            from insightface.app import FaceAnalysis
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Install InsightFace and the required ONNX Runtime "
                "dependencies to use ArcFace."
            ) from exc

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        self.app = FaceAnalysis(
            name=model_name,
            providers=providers,
        )

        self.app.prepare(
            ctx_id=0 if device == "cuda" else -1,
        )

        self.recognizer = self.app.models.get("recognition")

        if self.recognizer is None:
            raise RuntimeError(
                f"Model pack '{model_name}' has no recognition model."
            )

    def extract(self, face_crop: np.ndarray) -> np.ndarray:
        if (
            face_crop is None
            or not isinstance(face_crop, np.ndarray)
            or face_crop.size == 0
        ):
            raise ValueError("face_crop must be a non-empty NumPy array.")

        if face_crop.ndim != 3 or face_crop.shape[2] != 3:
            raise ValueError(
                "face_crop must be a three-channel image."
            )

        embedding = self.recognizer.get_feat([face_crop])

        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)

        if embedding.size == 0 or not np.isfinite(embedding).all():
            raise RuntimeError("ArcFace produced an invalid embedding.")

        norm = np.linalg.norm(embedding)

        if norm <= 1e-12:
            raise RuntimeError("ArcFace produced a zero-norm embedding.")

        return embedding / norm
