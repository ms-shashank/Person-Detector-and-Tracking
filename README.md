## README

### Person Detection & Tracking with YOLOv8 and DeepSORT

This project implements a person detection and tracking pipeline using the YOLOv8n object detection model and DeepSORT for multi-object tracking, along with OSNet for person re-identification. The code processes a video, detects persons in each frame, and tracks them across the video, ensuring that each person is uniquely identified even after temporary occlusion.

---

### Overview

The project aims to:

1. Detect persons in each frame using the **YOLOv8n** model.
2. Track the detected persons over time using **DeepSORT**.
3. Re-identify persons after occlusion or re-entry using the **OSNet** re-identification model.
4. Output the video with annotated bounding boxes and unique IDs for each person.

---

### Requirements

The following Python packages are required:

- `numpy`
- `scipy`
- `torch` / `torchvision`
- `ultralytics` (YOLOv8 model)
- `deep-sort-realtime`
- `opencv-python`
- `torchreid` (for OSNet re-identification)
- `gdown`

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

`torchreid` does not build reliably from PyPI on Python 3.12, so install it
from source:
```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

---

### Setup

1. **Download YOLOv8n Pretrained Model**:
   The pretrained YOLOv8n model is automatically downloaded by the `ultralytics` package. Make sure you are connected to the internet for this step.

2. **Download OSNet Pretrained Model**:
   The OSNet model is built into the `torchreid` package and is loaded directly in the script.

---

### Code Structure

- **`person_tracker.py`**:
  The pipeline. Loads the detector and the re-ID model, runs the frame loop, and
  draws the annotated output.

- **`reid.py`**:
  The identity gallery and box geometry. Deliberately depends on NumPy only, so
  the matching logic can be tested without OpenCV or torch.

- **`tests/`**:
  Unit tests for identity matching, appearance averaging, and the box helpers.

- **YOLOv8 detector**:
  Runs once per frame with `classes=[0]`, so only people are decoded.

- **DeepSORT tracker**:
  Associates boxes across frames, using the OSNet features as its appearance
  term instead of its own built-in embedder.

- **OSNet re-identification**:
  Produces one appearance vector per detected person. These feed both the
  tracker's association step and the gallery that outlives a lost track.

---

### Explanation of Key Components

#### 1. **YOLOv8 Model for Person Detection**
   - YOLOv8n is a state-of-the-art object detection model designed to detect multiple classes, including persons (class ID = 0).
   - The model processes each frame of the video to detect persons and output bounding boxes with confidence scores.

#### 2. **DeepSORT Tracker**
   - DeepSORT tracks objects (persons) by associating the bounding boxes from one frame to the next based on motion and appearance features.
   - It also handles occlusions and re-entry of the objects into the frame.

#### 3. **OSNet for Re-identification**
   - OSNet is a specialized network for person re-identification. It generates unique embeddings for each detected person.
   - These embeddings are used to compare and identify the same person even if the track is temporarily lost (e.g., due to occlusion).

#### 4. **Non-Maximum Suppression (NMS)**
   - After detecting multiple objects, NMS is applied to filter out overlapping detections and ensure that only the most confident detections remain.

#### 5. **Bounding Box and ID Assignment**
   - The tracker assigns a unique ID to each person detected in the video, and this ID persists across frames.
   - The IDs are displayed alongside bounding boxes, which are drawn around each detected person.

---

### Running the Code

1. **Ensure you have the video file**:
   Place your video in the `test_videos` directory, or pass its path with
   `--source`.

2. **Run the script**:
   ```bash
   python person_tracker.py --source test_videos/your_video.mp4 --output output_person_tracking.mp4
   ```

3. **View the output**:
   The script shows the video with a coloured box and an ID per tracked person,
   and writes the annotated video to `--output`. Press `q` to stop early. Pass
   `--no-display` on a headless machine or when batch processing.

4. **Run the tests**:
   ```bash
   python -m pytest tests -q
   ```

---

### Important Parameters

Every parameter below is a command line flag, so a run can be tuned without
editing the script.

- **`--conf 0.5`**:
  Minimum YOLO detection confidence. Lower it to keep partially occluded people;
  raise it to drop false positives.

- **`--max-age 30`**:
  Frames a track survives with no matching detection. Roughly one second at
  30 fps. Keep it near one second: a track that coasts much longer will
  re-attach to a different person walking through the same spot.

- **`--n-init 3`**:
  Consecutive detections before a track is confirmed and given an ID. Lower is
  faster to label, higher rejects flickering false positives.

- **`--nms-overlap 0.5`**:
  Overlap allowed by DeepSORT's NMS pass.

- **`--reid-threshold 0.35`**:
  Maximum cosine distance between a new appearance and a stored identity for
  them to count as the same person.

- **`--reid-margin 0.05`**:
  How much the best match must beat the runner-up. When two known people are
  equally close, the tracker starts a new identity instead of guessing.

- **`--reid-alpha 0.9`**:
  Weight of the stored appearance when folding in a new sighting. A high value
  keeps an identity stable through a single blurry or occluded frame.

- **`--reid-max-age 900`**:
  Frames an unclaimed identity stays in the gallery before it is forgotten.

---

### How Identity Is Kept Stable

Three things decide whether a person keeps their ID:

1. **The tracker sees the appearance features.** The OSNet vectors are passed
   into DeepSORT's association step, so a crossing or a brief occlusion is
   resolved on appearance as well as on motion.

2. **A gallery remembers people who left.** Each confirmed track owns an
   identity number. When that track dies and the person comes back, DeepSORT
   starts a new track, and the new track is matched against the gallery and
   inherits the original number.

3. **Matches must be unambiguous.** A new track only inherits an identity if it
   is within `--reid-threshold` *and* beats the next best candidate by
   `--reid-margin`. Identity numbers are never reused, and an identity that is
   already held by a live track can never be handed to a second one.

---

### Troubleshooting

1. **ID Switching**:
   Lower `--reid-threshold` if two different people are being merged into one
   ID; raise it if one person keeps getting a new ID. `--max-age` matters just
   as much: a value far above one second of video lets a dead track survive long
   enough to capture whoever walks into that spot next.

2. **Look-alike People Confused**:
   Increase `--reid-margin` so the best gallery match has to win by a clear
   margin before it is trusted.

3. **Overlapping Detections**:
   Lowering `--nms-overlap` removes redundant boxes more aggressively.

4. **Low FPS**:
   `--no-reid` skips OSNet entirely, and a smaller `--model` or a GPU
   (`--device 0`) both help.

---

### References

- YOLOv8: https://github.com/ultralytics/ultralytics
- DeepSORT: https://github.com/levan92/deep_sort_realtime
- OSNet: https://github.com/KaiyangZhou/deep-person-reid
