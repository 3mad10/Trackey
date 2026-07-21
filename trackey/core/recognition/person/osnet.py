import numpy as np
from typing import List

from trackey.core.interfaces.extractor import FeatureExtractor
from trackey.data.schemas.frame import Frame
from trackey.data.schemas.track import Track
from trackey.core.register import register_reid


@register_reid('osnet')
class OsNetReid(FeatureExtractor):
    def __init__(self, weights="osnet_ain_x1_0", device="cuda"):
        try:
            from torchreid.utils import FeatureExtractor
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Install torchreid from `https://github.com/KaiyangZhou/deep-person-reid`")
        
        self.extractor = FeatureExtractor(
            model_name=weights,
            device=device
        )


    def extract(self, tracks: List[Track], frame: Frame) -> List[np.ndarray]:
        if not tracks:
            return []
            
        cropped_frames = []
        for track in tracks:
            x1, y1, x2, y2 = track.bbox.to_pixel_xyxy(frame.width, frame.height)
            cropped_frame = frame.frame[y1:y2, x1:x2]
            cropped_frames.append(cropped_frame)
            
        embeddings = self.extractor(cropped_frames).cpu().numpy()
        return list(embeddings)

    def close(self):
        pass


if __name__=='__main__':
    import cv2
    image_path = "C:/Users/Mohamed Emad/OneDrive/Pictures/New York/20221203_203840.jpg"  # Replace with the actual path to your image
    image = cv2.imread(image_path)
    detector = YoloDetector()
    frame = Frame(frame = image, width=image.shape[1], height=image.shape[0])
    # print(detector.detect(frame=frame))