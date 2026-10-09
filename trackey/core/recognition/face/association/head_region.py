"""Associate face detections with person tracks using global assignment."""

import logging
from dataclasses import replace
from math import hypot
from typing import Dict, List, Tuple

from scipy.optimize import linear_sum_assignment

from trackey.core.interfaces.node import PipelineNode
from trackey.core.context import FrameContext
from trackey.data.schemas.detection import Detection
from trackey.data.schemas.track import Track  # Adjust to your actual Track module


logger = logging.getLogger(__name__)


class DetectionsAssociation(PipelineNode):
    """
    Associate face detections with person tracks.

    The node:
        1. Generates candidates using head-region containment.
        2. Rejects candidates below a confidence threshold.
        3. Scores the remaining candidates geometrically.
        4. Performs global one-to-one assignment.
        5. Leaves unmatched tracks unchanged.

    Assumes bounding-box coordinates are normalized to [0, 1].
    """

    # Confidence score = containment_weight * containment
    #                  + proximity_weight * proximity
    DEFAULT_CONTAINMENT_WEIGHT = 0.65
    DEFAULT_PROXIMITY_WEIGHT = 0.35

    def __init__(
        self,
        name: str,
        face_source: str = "face",
        head_region_ratio: float = 0.4,
        min_confidence: float = 0.25,
    ):
        super().__init__(name)

        if not 0.0 < head_region_ratio <= 1.0:
            raise ValueError("head_region_ratio must be in (0, 1].")

        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1].")

        self.face_source = face_source
        self.head_region_ratio = head_region_ratio
        self.min_confidence = min_confidence

    def process(self, ctx: FrameContext) -> FrameContext:
        faces = ctx.get_detections(self.face_source)
        tracks = ctx.tracks

        if not faces or not tracks:
            return ctx

        # Generate valid candidates and their geometric confidence scores.
        candidates = self._generate_candidates(tracks, faces)

        if not candidates:
            return ctx

        # Assign each track at most one face and each face at most one track.
        assignments = self._global_assignment(
            num_tracks=len(tracks),
            num_faces=len(faces),
            candidates=candidates,
        )

        if not assignments:
            return ctx

        # Update tracks immutably. Unmatched tracks remain unchanged.
        updated_tracks = list(tracks)

        for track_idx, face_idx in assignments.items():
            track = tracks[track_idx]
            face = faces[face_idx]

            updated_tracks[track_idx] = replace(
                track,
                face_bbox=face.bbox,
            )

        logger.debug(
            "Face association: %d tracks, %d faces, %d assignments",
            len(tracks),
            len(faces),
            len(assignments),
        )

        return ctx.with_tracks(updated_tracks)

    def _generate_candidates(
        self,
        tracks: List[Track],
        faces: List[Detection],
    ) -> Dict[Tuple[int, int], float]:
        """
        Generate candidate (track_index, face_index) pairs.

        Only faces whose centers fall inside a track's head region
        are considered. Candidates below min_confidence are discarded.
        """
        candidates: Dict[Tuple[int, int], float] = {}

        for track_idx, track in enumerate(tracks):
            if track.bbox is None:
                continue

            head_region = self._get_head_region(track.bbox)

            for face_idx, face in enumerate(faces):
                if face.bbox is None:
                    continue

                if not self._face_center_inside_region(
                    face.bbox,
                    head_region,
                ):
                    continue

                confidence = self._geometric_confidence(
                    face.bbox,
                    head_region,
                )

                if confidence < self.min_confidence:
                    continue

                candidates[(track_idx, face_idx)] = confidence

        return candidates

    def _get_head_region(self, track_bbox) -> Tuple[float, float, float, float]:
        """
        Return the person's upper-body bounding region as (x1, y1, x2, y2).

        The region extends from the top of the person bounding box
        to head_region_ratio * person_height.
        """
        x1 = track_bbox.cx - track_bbox.w / 2.0
        x2 = track_bbox.cx + track_bbox.w / 2.0

        y1 = track_bbox.cy - track_bbox.h / 2.0
        y2 = y1 + track_bbox.h * self.head_region_ratio

        # Clip the region to normalized frame coordinates.
        return (
            max(0.0, min(1.0, x1)),
            max(0.0, min(1.0, y1)),
            max(0.0, min(1.0, x2)),
            max(0.0, min(1.0, y2)),
        )

    @staticmethod
    def _face_center_inside_region(
        face_bbox,
        region: Tuple[float, float, float, float],
    ) -> bool:
        """Check whether the face center lies inside the head region."""
        x1, y1, x2, y2 = region

        return (
            x1 <= face_bbox.cx <= x2
            and y1 <= face_bbox.cy <= y2
        )

    @staticmethod
    def _geometric_confidence(
        face_bbox,
        region: Tuple[float, float, float, float],
    ) -> float:
        """
        Compute a geometric confidence score in [0, 1].

        Containment:
            Fraction of the face bounding-box area overlapping the
            head region.

        Proximity:
            How close the face center is to the center of the head region.

        The score measures geometric plausibility, not a calibrated
        probability that the face belongs to the person.
        """
        x1, y1, x2, y2 = region

        face_x1 = face_bbox.cx - face_bbox.w / 2.0
        face_y1 = face_bbox.cy - face_bbox.h / 2.0
        face_x2 = face_bbox.cx + face_bbox.w / 2.0
        face_y2 = face_bbox.cy + face_bbox.h / 2.0

        intersection_x1 = max(x1, face_x1)
        intersection_y1 = max(y1, face_y1)
        intersection_x2 = min(x2, face_x2)
        intersection_y2 = min(y2, face_y2)

        intersection_w = max(0.0, intersection_x2 - intersection_x1)
        intersection_h = max(0.0, intersection_y2 - intersection_y1)
        intersection_area = intersection_w * intersection_h

        face_area = max(0.0, face_bbox.w) * max(0.0, face_bbox.h)

        if face_area <= 0.0:
            return 0.0

        containment = min(1.0, intersection_area / face_area)

        region_cx = (x1 + x2) / 2.0
        region_cy = (y1 + y2) / 2.0

        distance = hypot(
            face_bbox.cx - region_cx,
            face_bbox.cy - region_cy,
        )

        # Normalize distance by the region's half-diagonal.
        half_diagonal = hypot(
            (x2 - x1) / 2.0,
            (y2 - y1) / 2.0,
        )

        if half_diagonal <= 0.0:
            proximity = 0.0
        else:
            proximity = max(0.0, 1.0 - distance / half_diagonal)

        confidence = (
            DetectionsAssociation.DEFAULT_CONTAINMENT_WEIGHT * containment
            + DetectionsAssociation.DEFAULT_PROXIMITY_WEIGHT * proximity
        )

        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _global_assignment(
        num_tracks: int,
        num_faces: int,
        candidates: Dict[Tuple[int, int], float],
    ) -> Dict[int, int]:
        """
        Maximize total geometric confidence with one-to-one assignment.

        Each track gets a dummy unmatched option with score zero.
        Invalid track-face pairs receive a prohibitive cost.

        Returns:
            {track_index: face_index}
        """
        if num_tracks == 0 or num_faces == 0 or not candidates:
            return {}

        # Add one dummy column per track so every track can remain unmatched.
        # Real face columns are shared, enforcing one-to-one matching.
        unmatched_score = 0.0
        invalid_cost = 1e6

        # scipy.optimize.linear_sum_assignment minimizes its cost matrix.
        cost = [
            [invalid_cost] * (num_faces + num_tracks)
            for _ in range(num_tracks)
        ]

        # Valid candidate costs are negative because lower costs are preferred.
        for (track_idx, face_idx), confidence in candidates.items():
            cost[track_idx][face_idx] = -confidence

        # Any track can use a dummy column to remain unmatched.
        for track_idx in range(num_tracks):
            cost[track_idx][num_faces + track_idx] = -unmatched_score

        row_indices, col_indices = linear_sum_assignment(cost)

        assignments: Dict[int, int] = {}

        for track_idx, col_idx in zip(row_indices, col_indices):
            # Dummy columns represent unmatched tracks.
            if col_idx >= num_faces:
                continue

            # Only accept explicitly generated candidate pairs.
            if (track_idx, col_idx) not in candidates:
                continue

            assignments[track_idx] = col_idx

        return assignments

    def get_inputs(self) -> List[str]:
        return ["tracks", "detections"]

    def get_outputs(self) -> List[str]:
        return ["tracks"]