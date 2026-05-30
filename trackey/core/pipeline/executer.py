import time
import logging
from collections import defaultdict, deque
from dataclasses import replace
from typing import Dict, List, Set, Optional

from trackey.core.interfaces.node import PipelineNode
from trackey.core.context import FrameContext
from trackey.core.pipeline.edge import Edge
from trackey.core.pipeline.constants import SKIP_BRANCH

logger = logging.getLogger(__name__)


class PipelineNodeError(Exception):
    def __init__(self, node_name: str, cause: Exception):
        self.node_name = node_name
        self.cause     = cause
        super().__init__(f"Node '{node_name}' failed: {cause}")


class PipelineExecutor:
    def __init__(self, nodes: List[PipelineNode], edges: List[Edge]):
        self.nodes: Dict[str, PipelineNode] = {n.name: n for n in nodes}
        self.out:   Dict[str, List[str]]    = defaultdict(list)
        self.in_:   Dict[str, List[str]]    = defaultdict(list)

        for edge in edges:
            self.out[edge.source].append(edge.target)
            self.in_[edge.target].append(edge.source)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(self, ctx: FrameContext) -> FrameContext:
        results:  Dict[str, FrameContext] = {}
        skipped:  Set[str]               = set()

        in_count: Dict[str, int] = {
            name: len(self.in_.get(name, []))
            for name in self.nodes
        }

        # start with source nodes — no parents
        queue = deque(
            name for name, count in in_count.items()
            if count == 0
        )

        while queue:
            node_name = queue.popleft()

            if node_name in skipped:
                continue

            input_ctx = self._resolve_input(node_name, results, ctx)
            if input_ctx is None:
                skipped.add(node_name)
                continue

            # execute with timing
            start = time.monotonic()
            try:
                out_ctx = self.nodes[node_name].process(input_ctx)
            except Exception as e:
                raise PipelineNodeError(node_name, e) from e

            elapsed = time.monotonic() - start
            out_ctx = replace(
                out_ctx,
                metadata={
                    **out_ctx.metadata,
                    f"latency.{node_name}": round(elapsed * 1000, 2)
                }
            )

            results[node_name] = out_ctx

            # handle branch routing
            if out_ctx.active_branch:
                active = out_ctx.active_branch
                if active == SKIP_BRANCH:
                    skipped.update(self.out.get(node_name, []))
                    next_nodes = []
                else:
                    inactive = [
                        c for c in self.out.get(node_name, [])
                        if c != active
                    ]
                    skipped.update(inactive)
                    next_nodes = [active]
                results[node_name] = replace(out_ctx, active_branch=None)
            else:
                next_nodes = [
                    c for c in self.out.get(node_name, [])
                    if c not in skipped
                ]
                results[node_name] = out_ctx

            # enqueue children whose all parents are done
            for child in next_nodes:
                parents_done = all(
                    p in results or p in skipped
                    for p in self.in_.get(child, [])
                )
                if parents_done:
                    queue.append(child)

        if skipped:
            logger.debug(f"[PipelineExecutor] Skipped: {skipped}")

        return self._final_context(results, ctx)

    # ------------------------------------------------------------------ #
    # Input resolution                                                     #
    # ------------------------------------------------------------------ #

    def _resolve_input(self,
                        node_name: str,
                        results: Dict[str, FrameContext],
                        initial_ctx: FrameContext) -> Optional[FrameContext]:
        parents = self.in_.get(node_name, [])

        if not parents:
            return initial_ctx

        active_ctxs = [
            results[p] for p in parents
            if p in results
        ]

        if not active_ctxs:
            return None

        if len(active_ctxs) == 1:
            return active_ctxs[0]

        return self._merge(active_ctxs)

    def _merge(self, ctxs: List[FrameContext]) -> FrameContext:
        base = ctxs[0]
        return replace(
            base,
            detections=[d for c in ctxs for d in c.detections],
            tracks=[t for c in ctxs for t in c.tracks],
            events=[e for c in ctxs for e in c.events],
            analytics={
                k: v for c in ctxs
                for k, v in c.analytics.items()
            },
            metadata={
                k: v for c in ctxs
                for k, v in c.metadata.items()
            },
        )

    # ------------------------------------------------------------------ #
    # Final context                                                        #
    # ------------------------------------------------------------------ #

    def _final_context(self,
                    results: Dict[str, FrameContext],
                    fallback: FrameContext) -> FrameContext:
        if not results:
            return fallback

        # terminal = ran AND has no children that also ran
        terminals = [
            name for name in results
            if not any(
                child in results
                for child in self.out.get(name, [])
            )
        ]

        if not terminals:
            return fallback
        if len(terminals) == 1:
            return results[terminals[0]]
        return self._merge([results[t] for t in terminals])