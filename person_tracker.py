"""Person detection and tracking with YOLOv8, DeepSORT and OSNet.

The pipeline detects people with YOLOv8, associates them across frames with
DeepSORT, and keeps identity numbers stable across occlusions by matching
OSNet appearance features against a gallery of everyone seen so far.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from reid import (
    DEFAULT_EMA_ALPHA,
    DEFAULT_MATCH_MARGIN,
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_MAX_AGE,
    IdentityGallery,
    best_detection_index,
    clamp_box,
    l2_normalize,
)

PERSON_CLASS_ID = 0
DEFAULT_SOURCE = (
    "test_videos/How+to+Do+Play+Therapy+_+Building+a+Growth+Mindset+Role+Play.mp4"
)
DEFAULT_OUTPUT = "output_person_tracking.mp4"

# Re-ID expects portrait crops, so the model's native input is 256 tall x 128 wide.
REID_INPUT_SIZE = (256, 128)

_PALETTE = [
    (56, 199, 255),
    (255, 128, 56),
    (120, 255, 120),
    (255, 96, 200),
    (180, 160, 255),
    (90, 220, 220),
    (255, 200, 90),
    (150, 150, 150),
]


def identity_colour(identity: int) -> Tuple[int, int, int]:
    """Deterministic colour per identity so a person keeps their colour."""
    return _PALETTE[(int(identity) - 1) % len(_PALETTE)]


def normalise_device(name: str) -> torch.device:
    """Accept 'cpu', 'cuda', 'cuda:0' or a bare index like '0'."""
    name = str(name).strip()
    if name.isdigit():
        return torch.device(f"cuda:{name}")
    return torch.device(name)


class OSNetEmbedder:
    """Turns person crops into L2-normalised OSNet appearance vectors."""

    def __init__(self, model_name: str, device: torch.device) -> None:
        try:
            import torchreid
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise ImportError(
                "torchreid is required for re-identification. Install it with\n"
                "  pip install git+https://github.com/KaiyangZhou/deep-person-reid.git\n"
                "or run with --no-reid to track without appearance matching."
            ) from exc

        self.device = device
        self.model = torchreid.models.build_model(
            name=model_name, num_classes=1000, pretrained=True
        )
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(REID_INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, crops_bgr: Sequence[np.ndarray]) -> List[np.ndarray]:
        if not crops_bgr:
            return []

        # OpenCV hands us BGR; the ImageNet statistics above expect RGB, so
        # feeding the raw crop swaps the red and blue channels and blurs the
        # features. Convert per crop rather than per frame.
        batch = torch.stack(
            [
                self.transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                for crop in crops_bgr
            ]
        ).to(self.device)

        with torch.inference_mode():
            try:
                # The classifier logits are not a metric space; the pooled
                # feature vector is what OSNet was trained to embed.
                features = self.model(batch, return_feature=True)
            except TypeError:  # pragma: no cover - older torchreid builds
                features = self.model(batch)

        return [
            l2_normalize(vector) for vector in features.float().cpu().numpy()
        ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and track people in a video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Video path or camera index.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Annotated video to write.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics detector weights.")
    parser.add_argument("--reid-model", default="osnet_x0_25", help="torchreid model name.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="'cpu', 'cuda', 'cuda:0' or an index like '0'.",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Detector confidence floor.")
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Frames a lost track survives. Large values let a dead track steal a new person.",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=3,
        help="Consecutive detections before a track is confirmed.",
    )
    parser.add_argument("--nms-overlap", type=float, default=0.5, help="DeepSORT NMS overlap.")
    parser.add_argument(
        "--reid-threshold",
        type=float,
        default=DEFAULT_MATCH_THRESHOLD,
        help="Maximum cosine distance for a re-identification match.",
    )
    parser.add_argument(
        "--reid-margin",
        type=float,
        default=DEFAULT_MATCH_MARGIN,
        help="How far the best match must beat the runner-up before it is trusted.",
    )
    parser.add_argument(
        "--reid-alpha",
        type=float,
        default=DEFAULT_EMA_ALPHA,
        help="Weight of the stored appearance when folding in a new sighting.",
    )
    parser.add_argument(
        "--reid-max-age",
        type=int,
        default=DEFAULT_MAX_AGE,
        help="Frames an unclaimed identity is remembered.",
    )
    parser.add_argument(
        "--no-reid",
        action="store_true",
        help="Skip OSNet entirely and rely on DeepSORT's own appearance model.",
    )
    parser.add_argument("--no-display", action="store_true", help="Write the video without a window.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = normalise_device(args.device)

    detector = YOLO(args.model)
    detector.to(device)

    embedder = None
    if not args.no_reid:
        try:
            embedder = OSNetEmbedder(args.reid_model, device)
        except Exception as exc:
            print(f"[warn] re-identification disabled: {exc}", file=sys.stderr)

    gallery = (
        IdentityGallery(
            match_threshold=args.reid_threshold,
            match_margin=args.reid_margin,
            ema_alpha=args.reid_alpha,
            max_age=args.reid_max_age,
        )
        if embedder is not None
        else None
    )

    tracker = DeepSort(
        max_age=args.max_age,
        n_init=args.n_init,
        nms_max_overlap=args.nms_overlap,
        embedder_gpu=device.type == "cuda",
    )

    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise SystemExit(f"Video source reported an unusable size: {width}x{height}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0 or not np.isfinite(source_fps):
        source_fps = 25.0

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Could not open video writer for: {args.output}")

    # DeepSORT track id -> stable identity number shown to the viewer.
    display_ids: dict[int, int] = {}
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            started = time.perf_counter()

            detections: list = []
            boxes_ltrb: list = []
            crops: list = []

            results = detector.predict(
                frame,
                classes=[PERSON_CLASS_ID],
                conf=args.conf,
                verbose=False,
            )[0]

            for x1, y1, x2, y2, confidence, _ in results.boxes.data.tolist():
                left, top, right, bottom = clamp_box(x1, y1, x2, y2, width, height)
                if right - left < 2 or bottom - top < 2:
                    continue

                boxes_ltrb.append([left, top, right, bottom])
                detections.append(
                    [[left, top, right - left, bottom - top], float(confidence), PERSON_CLASS_ID]
                )
                crops.append(frame[top:bottom, left:right])

            # Hand the OSNet features to the tracker itself. They are the whole
            # point of computing them: without this the association step falls
            # back to DeepSORT's own embedder and the re-ID features only ever
            # feed the gallery, so they cannot prevent a swap at the crossing.
            # Exactly one embedding per detection, in detection order -- the old
            # code skipped the append for unreadable crops, which silently
            # shifted every later embedding onto the wrong person.
            if embedder is not None:
                embeddings = embedder(crops)
                try:
                    tracks = tracker.update_tracks(
                        detections, embeds=embeddings, frame=frame
                    )
                except TypeError:  # pragma: no cover - older deep-sort-realtime
                    tracks = tracker.update_tracks(detections, frame=frame)
            else:
                embeddings = []
                tracks = tracker.update_tracks(detections, frame=frame)

            confirmed = [track for track in tracks if track.is_confirmed()]

            # Appearance for tracks that were matched to a detection this frame.
            # A coasting track has no fresh crop, so it contributes nothing to
            # the gallery instead of reusing someone else's features.
            fresh: dict[int, np.ndarray] = {}
            if embedder is not None:
                for track in confirmed:
                    if track.time_since_update != 0:
                        continue
                    index = best_detection_index(track.to_ltrb(), boxes_ltrb)
                    if index is not None:
                        fresh[track.track_id] = embeddings[index]

            # Claim every identity already owned by a live track first, so a new
            # track can never be handed an identity someone else is using.
            claimed = {
                display_ids[track.track_id]
                for track in confirmed
                if track.track_id in display_ids
            }

            for track in confirmed:
                track_id = track.track_id
                embedding = fresh.get(track_id)

                if track_id in display_ids:
                    if embedding is not None and gallery is not None:
                        gallery.observe(display_ids[track_id], embedding)
                    continue

                if gallery is None or embedding is None:
                    continue

                identity, _ = gallery.match(embedding, exclude=claimed)
                if identity is None:
                    identity = gallery.create(embedding)
                else:
                    gallery.observe(identity, embedding)

                display_ids[track_id] = identity
                claimed.add(identity)

            if gallery is not None:
                gallery.end_frame(claimed)

            for track in confirmed:
                left, top, right, bottom = (int(v) for v in track.to_ltrb())
                identity = display_ids.get(track.track_id, track.track_id)
                colour = identity_colour(identity)
                cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)
                cv2.putText(
                    frame,
                    f"ID {identity}",
                    (left, max(top - 10, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    colour,
                    2,
                )

            elapsed = time.perf_counter() - started
            cv2.putText(
                frame,
                f"FPS: {1.0 / elapsed:.2f}" if elapsed > 0 else "FPS: -",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

            writer.write(frame)

            if not args.no_display:
                cv2.imshow("Person Detection & Tracking (OSNet)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Processed {frame_index} frames -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
