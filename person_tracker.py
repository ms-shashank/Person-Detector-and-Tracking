import cv2
import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torchreid
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine

# Load YOLOv8n model
yolo_model = YOLO("yolov8n.pt")

# OSNet re-identification model
osnet_model = torchreid.models.build_model(
    name='osnet_x0_25', 
    num_classes=1000, 
    pretrained=True
)
osnet_model.eval() 

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


tracker = DeepSort(
    max_age=200, 
    n_init=5, 
    nms_max_overlap=0.5
)

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

video_cap = cv2.VideoCapture("test_videos/How+to+Do+Play+Therapy+_+Building+a+Growth+Mindset+Role+Play.mp4")

frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter("output_person_tracking_7.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

track_embedding_history = {}
last_seen = {}
REIDENTIFY_DELAY_FRAMES = 30

while video_cap.isOpened():
    ret, frame = video_cap.read()
    if not ret:
        break

    start_time = datetime.datetime.now()

    
    results = yolo_model(frame)[0]
    detections = []
    embeddings = []
    for data in results.boxes.data.tolist():
        confidence = data[4]
        if float(confidence) >= CONFIDENCE_THRESHOLD:
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])
            # Only track persons 
            if class_id == 0:
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                detections.append([bbox, confidence, class_id])
                
                person_img = frame[ymin:ymax, xmin:xmax]
                if person_img.size > 0:
                    person_img = transform(person_img).unsqueeze(0)
                    with torch.no_grad():
                        embedding = osnet_model(person_img)
                    embeddings.append(embedding.squeeze(0).cpu().numpy())

    tracks = tracker.update_tracks(detections, frame=frame)

    for i, track in enumerate(tracks):
        if not track.is_confirmed() or track.det_class != 0:  
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()  
        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

        if i < len(embeddings): 
            current_embedding = embeddings[i]
            if track_id not in track_embedding_history:
                if track_id in last_seen and (video_cap.get(cv2.CAP_PROP_POS_FRAMES) - last_seen[track_id]) < REIDENTIFY_DELAY_FRAMES:
                    continue  
                
                for past_id, past_embedding in track_embedding_history.items():
                    distance = cosine(current_embedding, past_embedding)
                    if distance < 0.25:
                        track_id = past_id
                        break

            track_embedding_history[track_id] = current_embedding
            last_seen[track_id] = video_cap.get(cv2.CAP_PROP_POS_FRAMES)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.putText(frame, f"ID: {track_id}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

    # Calculate and display FPS
    end_time = datetime.datetime.now()
    fps_text = f"FPS: {1 / (end_time - start_time).total_seconds():.2f}"
    cv2.putText(frame, fps_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Person Detection & Tracking (OSNet)", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
out.release()
cv2.destroyAllWindows()
