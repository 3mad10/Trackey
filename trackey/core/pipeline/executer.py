from trackey.core.pipeline.nodes import PipelineNode


class PipelineExecutor:
    def __init__(self, nodes: list[PipelineNode]):
        self.nodes = nodes

    def run(self, frame):
        data = {"frame": frame}

        for node in self.nodes:
            data = node[1].process(data)

        return data
