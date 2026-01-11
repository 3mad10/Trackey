from typing import Optional
from trackey.core.io.input import *
from trackey.core.io.output.viewer import *
from trackey.core.interfaces import *
from trackey.core.pipeline import PipelineExecutor
from trackey.core.factories.pipeline import PipelinBuilder


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

                result = self.pipeline.run(frame)
                if self.viewer:
                    self.viewer.show(
                        frame=result.get("frame"),
                        tracks=result.get("tracks", [])
                    )
        except KeyboardInterrupt:
            print("[Engine] Engine Stopped by user")
        if self.viewer:
            self.viewer.close()
        self.source.release()


if __name__ == '__main__':
    from trackey.core.io.input import CameraSource
    from trackey.core.io.output.viewer import OpenCVViewer
    from trackey.core.detectors import YoloDetector
    from trackey.core.trackers import DeepSortTracker
    pipeline_builder = PipelinBuilder(cfg_path="base_pipeline.yaml")
    pipeline = PipelineExecutor(pipeline_builder.build())
    engine = Engine(source=CameraSource(device_id=0), pipeline=pipeline, viewer=OpenCVViewer())
    engine.run()
