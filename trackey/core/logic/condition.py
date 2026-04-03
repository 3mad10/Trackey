import logging
from typing import Any

from trackey.core.interfaces.node import PipelineNode
from trackey.core.logic.path import PathExtractor
from trackey.core.context import FrameContext


logger = logging.getLogger(__name__)


class ConditionalNode(PipelineNode):
    """
    If/Then/Else logic node
    
    Example:
        If person count > 10:
            Execute alert node
        Else:
            Skip
    """
    def __init__(self,
                 name: str,
                 path: str,
                 operator: str,
                 value: Any,
                 event_name: str):
        super().__init__(name)
        self.extractor = PathExtractor(path)
        self.operator = operator
        self.threshold = value
        self.event_name = event_name

    def process(self, ctx: FrameContext) -> FrameContext:
        extracted = self.extractor.extract(ctx)
        if extracted is None:
            return ctx
        if self._evaluate(extracted):
            ctx.triggered_conditions.add(self.event_name)
        return ctx
    
    def _evaluate(self, value: Any) -> bool:
        ops = {
            "gt": lambda a, b: a > b,
            "lt": lambda a, b: a < b,
            "eq": lambda a, b: a == b,
            "gte": lambda a, b: a >= b,
            "lte": lambda a, b: a <= b,
        }
        op = ops.get(self.operator)
        if not op:
            logger.error(f"[ConditionalNode][{self.name}] Unknown operator: {self.operator}")
            return False
        return op(value, self.threshold)