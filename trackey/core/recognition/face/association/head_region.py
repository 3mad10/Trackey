




class DetectionsAssociation:
    """
    Associates independently-detected faces (whole-frame detector
    output, source="face" in ctx.detections) with person tracks
    using geometric containment.

    A face is considered to belong to a track if the face bbox
    center falls within the track's bbox, weighted toward the
    upper portion of the body (where heads are).

    Use only with high track density where per-track face detection
    (RetinaFaceCroppedNode) would be too expensive. Carries real
    failure modes in crowded/occluded scenes — validate against
    real footage before production use.
    """

    def __init__(self,
                 name:               str,
                 face_source:        str = "face",
                 head_region_ratio:  float = 0.4):
        """
        Args:
            face_source:       Key under ctx.detections where the
                                whole-frame face detector wrote results.
            head_region_ratio:  Fraction of the track bbox height (from
                                 the top) considered the "head region".
                                 A face must fall within this band to
                                 be associated. 0.4 means top 40%.
        """
        super().__init__(name)
        self.face_source       = face_source
        self.head_region_ratio = head_region_ratio

    def process(self, ctx: FrameContext) -> FrameContext:
        faces = ctx.get_detections(self.face_source)
        if not faces:
            return ctx

        updated_tracks = [
            self._associate(track, faces)
            for track in ctx.tracks
        ]
        return ctx.with_tracks(updated_tracks)

    def _associate(self, track: Track,
                   faces: List[Detection]) -> Track:
        if track.bbox is None:
            return track

        best_face = None
        best_score = 0.0

        for face in faces:
            if face.bbox is None:
                continue
            if not self._face_in_head_region(face.bbox, track.bbox):
                continue

            score = self._containment_score(face.bbox, track.bbox)
            if score > best_score:
                best_score = score
                best_face  = face

        if best_face is None:
            return track

        return track.model_copy(update={"face_bbox": best_face.bbox})

    def _face_in_head_region(self, face_bbox: BoundingBox,
                              track_bbox: BoundingBox) -> bool:
        tx1 = track_bbox.cx - track_bbox.w / 2
        tx2 = track_bbox.cx + track_bbox.w / 2
        ty1 = track_bbox.cy - track_bbox.h / 2
        ty2 = ty1 + track_bbox.h * self.head_region_ratio

        in_x = tx1 <= face_bbox.cx <= tx2
        in_y = ty1 <= face_bbox.cy <= ty2
        return in_x and in_y

    def _containment_score(self, face_bbox: BoundingBox,
                            track_bbox: BoundingBox) -> float:
        """Closer to top-center of the track bbox scores higher."""
        head_center_x = track_bbox.cx
        head_center_y = (track_bbox.cy - track_bbox.h / 2) + (
            track_bbox.h * self.head_region_ratio / 2
        )
        dx = abs(face_bbox.cx - head_center_x)
        dy = abs(face_bbox.cy - head_center_y)
        return 1.0 / (1.0 + dx + dy)

    def get_inputs(self) -> List[str]:
        return ["tracks", "detections"]

    def get_outputs(self) -> List[str]:
        return ["tracks"]