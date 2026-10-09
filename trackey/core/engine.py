import logging
import time
import threading
import queue
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

logger = logging.getLogger(__name__)
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
        stores          = StoreBuilder(cfg_path).build()
        nodes, edges    = PipelineBuilder(cfg_path, scene, event_bus, stores).build()
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
        
        self.frame_queue = queue.Queue(maxsize=30)
        self.running = True
        
        capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        capture_thread.start()
        
        try:
            while self.running:
                self._process_frame()
        except KeyboardInterrupt:
            logger.info("[Engine] Stopped by user")
            self.running = False
        except StopIteration:
            logger.info("[Engine] Source exhausted")
            self.running = False
        finally:
            self._close_all()
            
    def _capture_loop(self):
        while self.running:
            if self._buffered_frame is not None:
                frame = self._buffered_frame
                self._buffered_frame = None
            else:
                frame = self.source.read()
                
            if frame is None:
                self.running = False
                break
                
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame, block=False)
                except queue.Empty:
                    pass

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
            return

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
        try:
            return self.frame_queue.get(timeout=0.5)
        except queue.Empty:
            if not self.running:
                raise StopIteration
            return None

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
            sink.close()




