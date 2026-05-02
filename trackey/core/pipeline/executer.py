import time
import logging
from typing import List, Dict, Set, Optional
from collections import defaultdict, deque
from dataclasses import replace

from trackey.core.interfaces.node   import PipelineNode
from trackey.core.context           import FrameContext
from trackey.core.pipeline.edge     import Edge
from trackey.core.utils.graph       import topological_sort

logger = logging.getLogger(__name__)


class PipelineExecutor:
    def __init__(self, nodes: List[PipelineNode], edges: List[Edge]):
        self.nodes:  Dict[str, PipelineNode] = {n.name: n for n in nodes}
        self.edges:  List[Edge]              = edges
        self._out:   Dict[str, List[Edge]]   = self._build_adjacency()
        self._in:    Dict[str, List[Edge]]   = self._build_reverse_adjacency()
        self.order:  List[str]               = topological_sort(list(self.nodes.keys()), self.edges)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def run(self, ctx: FrameContext) -> FrameContext:
        skipped: Set[str]                  = set()
        results: Dict[str, FrameContext]   = {} # Dict of node -> result ctx

        for node_name in self.order:
            if node_name in skipped:
                continue

            # resolve which context this node receives
            input_ctx = self._resolve_input(node_name, results, ctx)
            if input_ctx is None:
                # all incoming edges inactive
                skipped.update(
                    self._collect_descendants(node_name)
                )
                continue

            # execute with timing
            start = time.monotonic()
            try:
                result_ctx = self.nodes[node_name].process(input_ctx)
            except Exception as e:
                logger.error(
                    f"[PipelineExecutor] Node '{node_name}' failed: {e}",
                    exc_info=True
                )
                raise PipelineNodeError(node_name, e) from e

            elapsed = time.monotonic() - start
            result_ctx.metadata['execution_time'][node_name] = elapsed
            results[node_name] = result_ctx

            # handle switch/condition routing
            if result_ctx.active_branch is not None:
                inactive = self._collect_inactive_branches(
                    node_name, result_ctx.active_branch
                )
                skipped.update(inactive)
                result_ctx = replace(result_ctx, active_branch=None)

        return self._final_context(results, ctx)

    # ------------------------------------------------------------------ #
    # Graph construction                                                 #
    # ------------------------------------------------------------------ #

    def _build_adjacency(self) -> Dict[str, List[Edge]]:
        adj = defaultdict(list)
        for edge in self.edges:
            adj[edge.source].append(edge)
        return dict(adj)

    def _build_reverse_adjacency(self) -> Dict[str, List[Edge]]:
        rev = defaultdict(list)
        for edge in self.edges:
            rev[edge.target].append(edge)
        return dict(rev)

    def _topological_sort(self) -> List[str]:
        # Topological sort using kahn algorithm
        in_degree = {name: 0 for name in self.nodes}
        for edge in self.edges:
            in_degree[edge.target] += 1

        queue = deque(
            name for name, degree in in_degree.items()
            if degree == 0
        )
        order = []

        while queue:
            node_name = queue.popleft()
            order.append(node_name)
            for edge in self._out.get(node_name, []):
                in_degree[edge.target] -= 1
                if in_degree[edge.target] == 0:
                    queue.append(edge.target)

        if len(order) != len(self.nodes):
            cycle = self._find_cycle()
            raise ValueError(
                f"[PipelineExecutor] Cycle detected: {' → '.join(cycle)}"
            )

        return order

    # ------------------------------------------------------------------ #
    # Input resolution                                                     #
    # ------------------------------------------------------------------ #

    def _resolve_input(self, node_name: str,
                    results: Dict[str, FrameContext],
                    initial_ctx: FrameContext) -> Optional[FrameContext]:
        incoming = self._in.get(node_name, [])

        if not incoming:
            return initial_ctx

        # edges carry no logic — just check if source was skipped
        active_contexts = [
            results[edge.source]
            for edge in incoming
            if edge.source in results
            and edge.source not in self._skipped  # no condition evaluation
        ]

        if not active_contexts:
            return None

        if len(active_contexts) == 1:
            return active_contexts[0]

        return self._merge_contexts(active_contexts)

    def _merge_contexts(self,
                         contexts: List[FrameContext]) -> FrameContext:
        base = contexts[0]
        merged_detections = []
        merged_tracks     = []
        merged_analytics  = {}
        merged_events     = []
        metadata  = {}

        for ctx in contexts:
            merged_detections.extend(ctx.detections)
            merged_tracks.extend(ctx.tracks)
            merged_analytics.update(ctx.analytics)
            merged_events.extend(ctx.events)
            metadata.update(ctx.metadata)

        return replace(
            base,
            detections    = merged_detections,
            tracks        = merged_tracks,
            analytics     = merged_analytics,
            events        = merged_events,
            metadata      = metadata,
        )

    # ------------------------------------------------------------------ #
    # Branch routing                                                       #
    # ------------------------------------------------------------------ #

    def _collect_inactive_branches(self,
                                    switch_node: str,
                                    active_target: str) -> Set[str]:
        inactive_starts = [
            edge.target
            for edge in self._out.get(switch_node, [])
            if edge.target != active_target
        ]
        inactive = set()
        for start in inactive_starts:
            inactive.update(
                self._collect_descendants(start, stop_at=active_target)
            )
        return inactive

    def _collect_descendants(self,
                               start: str,
                               stop_at: Optional[str] = None) -> Set[str]:
        visited = set()
        queue   = deque([start])

        while queue:
            current = queue.popleft()
            if current == stop_at:
                continue
            if current in visited or current not in self.nodes:
                continue
            visited.add(current)
            for edge in self._out.get(current, []):
                queue.append(edge.target)

        return visited

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    def _final_context(self,
                        results: Dict[str, FrameContext],
                        fallback: FrameContext) -> FrameContext:
        if not results:
            return fallback
        # return context from last node in topological order
        for name in reversed(self.order):
            if name in results:
                return results[name]
        return fallback

    def _find_cycle(self) -> List[str]:
        visited   = set()
        rec_stack = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.append(node)
            for edge in self._out.get(node, []):
                if edge.target not in visited:
                    if dfs(edge.target):
                        return True
                elif edge.target in rec_stack:
                    idx = rec_stack.index(edge.target)
                    rec_stack.append(edge.target)
                    return True
            rec_stack.pop()
            return False

        for node in self.nodes:
            if node not in visited:
                if dfs(node):
                    return rec_stack
        return []


class PipelineNodeError(Exception):
    def __init__(self, node_name: str, cause: Exception):
        self.node_name = node_name
        self.cause     = cause
        super().__init__(f"Node '{node_name}' failed: {cause}")
