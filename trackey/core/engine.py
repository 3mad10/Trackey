import logging
import time
from typing import Optional, List

from trackey.core.io.input import *
from trackey.core.io.output.viewer import *
from trackey.core.interfaces import *
from trackey.core.pipeline import PipelineExecutor
from trackey.core.factories import *
from trackey.core.interfaces.sink import OutputSink
from trackey.core.interfaces.source import InputSource
from trackey.core.interfaces.renderer import Renderer
from trackey.data.schemas.pipeline import PipelineResult
from trackey.data.schemas.event import BaseEvent
from trackey.core.context import FrameContext
from trackey.data.schemas.frame import Frame



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
    def __init__(self,
                 source:              InputSource,
                 executor:            PipelineExecutor,
                 renderer:            Optional[Renderer],
                 sinks:               List[OutputSink],
                 event_bus:           Optional[BaseEvent] = None,
                 open_source_retries: int = 10):
        self.source              = source
        self.executor            = executor
        self.renderer            = renderer
        self.sinks               = sinks
        self.event_bus           = event_bus
        self.open_source_retries = open_source_retries
        self._frame_id           = 0
        self._buffered_frame: Optional[Frame] = None

    @classmethod
    def from_config(cls, cfg_path: str) -> "Engine":
        scene           = SceneBuilder(cfg_path).build()
        source          = SourceBuilder(cfg_path).build()
        event_bus       = EventBusBuilder(cfg_path).build()
        nodes, edges    = PipelineBuilder(cfg_path, scene, event_bus).build()
        renderer        = RendererBuilder(cfg_path, scene).build()
        sinks           = SinkBuilder(cfg_path).build()
        
        pipeline = PipelineExecutor(nodes, edges)

        return cls(
            source=source,
            executor=pipeline,
            renderer=renderer,
            sinks=sinks,
            event_bus=event_bus,
        )

    def run(self) -> None:
        self._open_all()
        self._initialize_renderer()
        try:
            while True:
                self._process_frame()
        except KeyboardInterrupt:
            logger.info("[Engine] Stopped by user")
        except StopIteration:
            logger.info("[Engine] Source exhausted")
        finally:
            self._close_all()
    
    def _initialize_renderer(self) -> None:
        if not self.renderer:
            return

        frame = None
        for attempt in range(self.open_source_retries):
            frame = self.source.read()
            if frame is not None:
                break
            logger.warning(
                f"[Engine] Failed to read first frame "
                f"(attempt {attempt + 1}/{self.open_source_retries})"
            )

        if frame is None:
            raise RuntimeError(
                f"[Engine] Could not read first frame after "
                f"{self.open_source_retries} attempts"
            )

        self.renderer.initialize(frame)
        self._buffered_frame = frame  # buffer first frame

    def _process_frame(self) -> None:
        frame = self._get_next_frame()
        if frame is None:
            raise StopIteration

        ctx = FrameContext(
            frame_id=self._frame_id,
            camera_id=self.source.camera_id,
            timestamp=time.monotonic(),
            frame=frame
        )

        ctx = self.executor.run(ctx)

        # render if renderer exists
        rendered = self.renderer.render(ctx) if self.renderer else None

        # build result — carries both rendered and raw frame
        result = PipelineResult.from_context(ctx, rendered=rendered)

        # all sinks get same result, use what they need
        for sink in self.sinks:
            sink.write(result)

        self._frame_id += 1

    def _get_next_frame(self) -> Optional[Frame]:
        # return buffered first frame if available
        if self._buffered_frame is not None:
            frame = self._buffered_frame
            self._buffered_frame = None
            return frame
        return self.source.read()

    def _open_all(self) -> None:
        if not self.source.open():
            raise RuntimeError(
                f"[Engine] Failed to open source"
            )
        for sink in self.sinks:
            if not sink.open():
                raise RuntimeError(
                    f"[Engine] Failed to open sink: {sink.__class__.__name__}"
                )

    def _close_all(self) -> None:
        if self.event_bus:
            self.event_bus.stop()
        self.source.release()
        for sink in self.sinks:
            sink.release()



if __name__ == '__main__':
    from trackey.plugins.io.camera import CameraSourcePlugin
    from trackey.plugins.io.video import VideoSourcePlugin
    from trackey.plugins.io.display import DisplaySinkPlugin
    from trackey.plugins.subscribers.mail import MailSubscriberPlugin
    from trackey.core.io.output.viewer import OpenCVViewer
    from trackey.core.detectors import YoloDetector
    from trackey.core.trackers import DeepSortTracker
    from trackey.core.analyzers import Counter
    from trackey.core.rendering.opencv_renderer import OpenCVRenderer
    # scene = SceneBuilder(cfg_path="base_pipeline.yaml").build()
    # pipeline_builder = PipelineBuilder(cfg_path="base_pipeline.yaml", scene=scene)
    # pipeline = PipelineExecutor(pipeline_builder.build())
    # engine = Engine(source=CameraSource(device_id=0),
    #                 pipeline=pipeline,
    #                 viewer=OpenCVViewer(),
    #                 renderer=Renderer(),
    #                 scene=scene
    #                 )
    engine = Engine.from_config("base_pipeline.yaml")
    engine.run()
