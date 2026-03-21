import logging
from typing import Dict, Callable, List, Any
from trackey.core.interfaces.node import PipelineNode

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
                 condition:Callable[[Dict], bool],
                 true_nodes:List[PipelineNode],
                 false_nodes:List[PipelineNode]):
        self.true_nodes = true_nodes or []
        self.false_nodes = false_nodes or []
        self.condition = condition
    def process(self, data: Dict) -> Dict:
        if self.condition(data):
            for node in self.true_nodes:
                data = node.process(data)
        else:
            for node in self.false_nodes:
                data = node.process(data)
        return data