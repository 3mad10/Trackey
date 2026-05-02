# trackey/core/pipeline/graph.py

from collections import defaultdict, deque
from typing import List, Dict
from trackey.core.pipeline.edge import Edge


def topological_sort(node_names: List[str], edges: List[Edge]) -> List[str]:
    in_degree = {name: 0 for name in node_names}
    out_map   = defaultdict(list)

    for edge in edges:
        in_degree[edge.target] += 1
        out_map[edge.source].append(edge.target)

    queue = deque(
        name for name, degree in in_degree.items()
        if degree == 0
    )
    order = []

    while queue:
        name = queue.popleft()
        order.append(name)
        for child in out_map[name]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != len(node_names):
        raise ValueError("Cycle detected in pipeline graph")

    return order