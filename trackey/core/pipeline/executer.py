import time

from trackey.core.interfaces.node import PipelineNode
from trackey.core.context import FrameContext


class PipelineExecutor:
    def __init__(self, nodes: list[PipelineNode]):
        self.nodes = nodes
        self.frame_id = 0

    def run(self, frame):
        ctx = FrameContext(frame=frame, timestamp=time.time())

        for node in self.nodes:
            start = time.time()
            ctx = node.process(ctx)
            ctx.metadata[node.name] = {
                "time_ms": (time.time()-start)*1000
            }
        return ctx
