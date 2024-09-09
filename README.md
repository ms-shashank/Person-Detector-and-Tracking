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
- `torch`
- `ultralytics` (YOLOv8 model)
- `deep-sort-realtime`
- `opencv-python`
- `torchreid` (for OSNet re-identification)
- `gdown`

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

---

### Setup

1. **Download YOLOv8n Pretrained Model**:
   The pretrained YOLOv8n model is automatically downloaded by the `ultralytics` package. Make sure you are connected to the internet for this step.

2. **Download OSNet Pretrained Model**:
   The OSNet model is built into the `torchreid` package and is loaded directly in the script.

---

### Code Structure

- **`yolo_model = YOLO("yolov8n.pt")`**: 
  Loads the YOLOv8n model, which is used to detect objects (particularly persons) in the video frames.
  
- **DeepSORT Initialization**: 
  The DeepSORT tracker is initialized to handle object tracking by associating bounding boxes across frames.

- **OSNet for Person Re-identification**: 
  OSNet is used to generate embeddings for each detected person. This allows the model to re-identify people after occlusions.

- **Main Loop**: 
  The code reads the video frame-by-frame, applies YOLO to detect persons, uses DeepSORT for tracking, and applies re-identification with OSNet to ensure consistent IDs throughout the video.

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

To run the project:

1. **Ensure you have the necessary video file**:
   Place your video in the `test_videos` directory or adjust the path to point to your video file.

2. **Run the script**:
   ```bash
   python person_tracking.py
   ```

3. **View the output**:
   The script will display the video with bounding boxes and unique IDs for each tracked person. It will also save the output video (`output_person_tracking.mp4`) to the current directory.

---

### Important Parameters

- **`CONFIDENCE_THRESHOLD = 0.5`**: 
  Minimum confidence for YOLO detections. Increase this value to reduce false positives.
  
- **`NMS_THRESHOLD = 0.3`**: 
  Threshold for Non-Maximum Suppression to handle overlapping bounding boxes.
  
- **`REIDENTIFY_DELAY_FRAMES = 30`**: 
  Maximum number of frames to wait before attempting re-identification of a lost person track.

- **DeepSORT Parameters**:
  - `max_age=200`: Maximum number of frames an object can remain undetected before its track is removed.
  - `n_init=5`: Minimum number of consecutive detections before an object is confirmed as being tracked.
  - `nms_max_overlap=0.5`: Maximum allowed overlap for NMS in the DeepSORT tracker.

---

### Troubleshooting

1. **ID Switching**:
   If you notice frequent switching of person IDs, try adjusting the `max_age`, `n_init`, and `nms_max_overlap` parameters of the DeepSORT tracker to improve tracking consistency.

2. **Overlapping Detections**:
   If overlapping detections are an issue, increasing the `NMS_THRESHOLD` can help eliminate redundant bounding boxes.

3. **Low FPS**:
   If the processing speed is low, consider reducing the video resolution or switching to a faster model (e.g., a lighter version of YOLOv8).

---

### References

- YOLOv8: https://github.com/ultralytics/ultralytics
- DeepSORT: https://github.com/levan92/deep_sort_realtime
- OSNet: https://github.com/KaiyangZhou/deep-person-reid
