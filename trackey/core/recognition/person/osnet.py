from typing import List

from trackey.core.interfaces.extractor import FeatureExtractor
from trackey.data.schemas.frame import Frame
from trackey.data.schemas.track import Track
from trackey.data.schemas.identity import Identity
from trackey.core.register import register_reid


@register_reid('osnet')
class OsNetReid(FeatureExtractor):
    def __init__(self, model_name="osnet_ain_x1_0", device="cuda"):
        try:
            from torchreid.utils import FeatureExtractor
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Install torchreid from `https://github.com/KaiyangZhou/deep-person-reid`")
        
        self.extractor = FeatureExtractor(
            model_name=model_name,
            device=device
        )


    def extract(self, tracks: List[Track], frame: Frame) -> List[Track]:
        for track in tracks:
            x1, y1, x2, y2 = track.bbox.to_pixel_xyxy(frame.width, frame.height)
            cropped_frame = frame.frame[x1:x2, y1:y2]
            embedding = self.extractor(cropped_frame)
            if track.identity:
                track.identity.add_embedding(embedding)
                print("added new embedding", embedding)
            else:
                track.identity = Identity()
                print("created new Identity", embedding)
        return tracks

    def close(self):
        pass


if __name__=='__main__':
    import cv2
    image_path = "C:/Users/Mohamed Emad/OneDrive/Pictures/New York/20221203_203840.jpg"  # Replace with the actual path to your image
    image = cv2.imread(image_path)
    detector = YoloDetector()
    frame = Frame(frame = image, width=image.shape[1], height=image.shape[0])
    # print(detector.detect(frame=frame))