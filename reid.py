"""Appearance identity gallery and geometry helpers used by the tracker.

Everything here depends on NumPy only -- no OpenCV, no torch -- so the logic
that decides *who is who* can be unit-tested without the inference stack.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Set, Tuple

import numpy as np

# A candidate is accepted only if it is this close to the stored appearance.
DEFAULT_MATCH_THRESHOLD = 0.35
# ...and only if it beats the runner-up by this much. Without a margin, a person
# who looks similar to two known people gets merged into whichever was seen
# first, which is the main cause of silent ID swaps.
DEFAULT_MATCH_MARGIN = 0.05
# Weight of the running average. High alpha keeps the appearance model stable
# so a single blurry or occluded frame cannot rewrite a person's identity.
DEFAULT_EMA_ALPHA = 0.9
# Frames an unclaimed identity is kept before it is forgotten.
DEFAULT_MAX_AGE = 900

# A track is only linked to a detection when their boxes overlap at least this
# much. Coasting tracks (no detection this frame) fall below it naturally.
MIN_IOU = 0.5


def l2_normalize(vector: Sequence[float]) -> np.ndarray:
    """Return ``vector`` scaled to unit length, or zeros if it is degenerate."""
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.zeros_like(array)
    return array / norm


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine distance in ``[0, 2]``; 0 means identical direction."""
    a = l2_normalize(left)
    b = l2_normalize(right)
    if not a.any() or not b.any():
        return 2.0
    return float(np.clip(1.0 - float(np.dot(a, b)), 0.0, 2.0))


def clamp_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """Clip a box to the frame so slicing it always yields a usable crop.

    Detector boxes can sit a pixel outside the image, and ``frame[y1:y2, x1:x2]``
    on such a box returns an empty array rather than the person.
    """
    left = max(0, min(int(round(x1)), width - 1))
    top = max(0, min(int(round(y1)), height - 1))
    right = max(left + 1, min(int(round(x2)), width))
    bottom = max(top + 1, min(int(round(y2)), height))
    return left, top, right, bottom


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)

    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0

    intersection = inter_w * inter_h
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def best_detection_index(
    track_box: Sequence[float],
    detection_boxes: Sequence[Sequence[float]],
    min_iou: float = MIN_IOU,
) -> Optional[int]:
    """Index of the detection that best explains ``track_box``, or ``None``.

    DeepSORT returns tracks in an order that has nothing to do with the order
    the detections were supplied in, and it also returns tracks that matched no
    detection at all. Looking an appearance vector up by list position therefore
    hands one person's features to another. Matching on box overlap instead ties
    each track to the detection it was actually built from.
    """
    best_index = None
    best_iou = min_iou
    for index, box in enumerate(detection_boxes):
        overlap = box_iou(track_box, box)
        if overlap >= best_iou:
            best_index = index
            best_iou = overlap
    return best_index


class IdentityGallery:
    """Maps appearance vectors to stable, never-reused identity numbers."""

    def __init__(
        self,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
        match_margin: float = DEFAULT_MATCH_MARGIN,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        max_age: int = DEFAULT_MAX_AGE,
    ) -> None:
        self.match_threshold = float(match_threshold)
        self.match_margin = float(match_margin)
        self.ema_alpha = float(ema_alpha)
        self.max_age = int(max_age)

        self._embeddings: dict[int, np.ndarray] = {}
        self._hits: dict[int, int] = {}
        self._age: dict[int, int] = {}
        self._next_id = 1

    def __len__(self) -> int:
        return len(self._embeddings)

    @property
    def ids(self) -> Set[int]:
        return set(self._embeddings)

    def create(self, embedding: Sequence[float]) -> int:
        """Register a new appearance and return its freshly allocated id."""
        identity = self._next_id
        self._next_id += 1
        self._embeddings[identity] = l2_normalize(embedding)
        self._hits[identity] = 1
        self._age[identity] = 0
        return identity

    def observe(self, identity: int, embedding: Sequence[float]) -> None:
        """Fold a new sighting into an identity's running appearance model."""
        observation = l2_normalize(embedding)
        if not observation.any():
            return

        current = self._embeddings.get(identity)
        if current is None:
            self._embeddings[identity] = observation
        else:
            blended = self.ema_alpha * current + (1.0 - self.ema_alpha) * observation
            self._embeddings[identity] = l2_normalize(blended)

        self._hits[identity] = self._hits.get(identity, 0) + 1
        self._age[identity] = 0

    def match(
        self,
        embedding: Sequence[float],
        exclude: Iterable[int] = (),
    ) -> Tuple[Optional[int], float]:
        """Return ``(identity, distance)`` for the best inactive candidate.

        The identity is ``None`` when nothing is close enough or when the two
        closest candidates are too similar to tell apart, in which case the
        caller should start a new identity rather than guess.
        """
        query = l2_normalize(embedding)
        if not query.any():
            return None, 2.0

        blocked = set(exclude)
        candidates = [
            (cosine_distance(query, reference), identity)
            for identity, reference in self._embeddings.items()
            if identity not in blocked
        ]
        if not candidates:
            return None, 2.0

        candidates.sort()
        best_distance, best_identity = candidates[0]
        if best_distance > self.match_threshold:
            return None, best_distance
        if len(candidates) > 1 and candidates[1][0] - best_distance < self.match_margin:
            return None, best_distance
        return best_identity, best_distance

    def end_frame(self, claimed: Iterable[int] = ()) -> None:
        """Age identities not claimed this frame and forget the stale ones."""
        live = set(claimed)
        for identity in list(self._embeddings):
            if identity in live:
                self._age[identity] = 0
                continue
            age = self._age.get(identity, 0) + 1
            self._age[identity] = age
            if age > self.max_age:
                del self._embeddings[identity]
                self._hits.pop(identity, None)
                self._age.pop(identity, None)
