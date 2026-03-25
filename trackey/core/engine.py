import logging
from typing import Optional

from trackey.core.io.input import *
from trackey.core.io.output.viewer import *
from trackey.core.interfaces import *
from trackey.core.pipeline import PipelineExecutor
from trackey.core.factories.pipeline import PipelinBuilder
from trackey.core.factories.scene import SceneBuilder
from trackey.data.schemas.pipeline import PipelineResult


logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
file_handler = logging.FileHandler("trackey.log")
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

class Engine:
    def __init__(
            self,
            source: InputSource,
            pipeline: PipelineExecutor,
            viewer: Optional[OutputViewer]
            ):
        self.source = source
        self.pipeline = pipeline
        self.viewer = viewer if viewer else None

    def run(self):
        self.source.open()
        if self.viewer:
            self.viewer.open()

        try:
            while True:
                frame = self.source.read()
                if frame is None:
                    break

                ctx = self.pipeline.run(frame)

                result = PipelineResult(
                    frame_id=ctx.frame_id,
                    timestamp=ctx.timestamp,
                    detections=ctx.detections,
                    tracks=ctx.tracks,
                    analytics=ctx.analytics,
                    metadata=ctx.metadata
                )
                if self.viewer:
                    self.viewer.show(frame, result)

        except KeyboardInterrupt:
            print("[Engine] Engine stopped by user")

        finally:
            if self.viewer:
                self.viewer.close()
            self.source.release()



if __name__ == '__main__':
    from trackey.core.io.input import CameraSource, VideoFileSource
    from trackey.core.io.output.viewer import OpenCVViewer
    from trackey.core.detectors import YoloDetector
    from trackey.core.trackers import DeepSortTracker
    from trackey.core.analyzers import Counter
    scene = SceneBuilder(cfg_path="base_pipeline.yaml").build()
    pipeline_builder = PipelinBuilder(cfg_path="base_pipeline.yaml", scene=scene)
    pipeline = PipelineExecutor(pipeline_builder.build())
    # engine = Engine(source=CameraSource(device_id=0),
    #                 pipeline=pipeline,
    #                 viewer=OpenCVViewer(scene=scene))
    engine = Engine(source=VideoFileSource("/home/mohamed-emad/Downloads/IP Camera Demo traffic car.mp4"),
                    pipeline=pipeline,
                    viewer=OpenCVViewer(scene=scene))
    engine.run()
