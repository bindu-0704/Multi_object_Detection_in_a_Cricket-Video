  

import os
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from sort import *

# --- Configuration ---
video_folder = r"C:\Users\moon\Downloads\detection\object_detection\video"  # Folder containing input videos
output = r"C:\Users\moon\Downloads\detection\object_detection\output"  # Folder to save output videos
model_path = r"C:\Users\moon\Downloads\detection\object_detection\best.pt"

count_notprocessed = 0

# Load YOLOv8 model
model = YOLO(model_path)

# Custom class names from the model
classNames = ['Bat', 'Gloves', 'Stump', 'Umpire', 'ball', 'player']

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=5, iou_threshold=0.5)

# Process all video files in the folder
for folder in os.listdir(video_folder):
    input_folder = os.path.join(video_folder, folder)
    if os.path.isdir(input_folder):
        # Create output folder inside the specified output directory
        output_folder = os.path.join(output, folder)
        os.makedirs(output_folder, exist_ok=True)

        for video in os.listdir(input_folder):
            video_path = os.path.join(input_folder, video)

            # Open the video file
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                count_notprocessed += 1
                print(f"Failed to open video: {video}")
                continue

            print(f"Processing: {video}")
            width = int(cap.get(3))
            height = int(cap.get(4))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Output path for the processed video
            output_path = os.path.join(output_folder, f"output_{video}")
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            # Track unique players
            unique_player_ids = set()

            while True:
                success, img = cap.read()
                if not success:
                    break

                results = model(img, stream=True)
                detections = np.empty((0, 5))
                labels_to_draw = []

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = round(float(box.conf[0]), 2)
                        cls = int(box.cls[0])
                        if cls >= len(classNames):
                            continue
                        class_name = classNames[cls]

                        if conf > 0.5:
                            if class_name == "player":
                                currentArray = np.array([x1, y1, x2, y2, conf])
                                detections = np.vstack((detections, currentArray))
                            labels_to_draw.append((x1, y1, x2, y2, class_name, conf))

                tracked_players = tracker.update(detections)

                for result in tracked_players:
                    x1, y1, x2, y2, Id = map(int, result)
                    w, h = x2 - x1, y2 - y1
                    unique_player_ids.add(Id)
                    cvzone.cornerRect(img, (x1, y1, w, h), l=15)
                    cvzone.putTextRect(img, f'Player {Id}', (x1, max(40, y1)), scale=0.8, thickness=1)

                for x1, y1, x2, y2, label, conf in labels_to_draw:
                    if label != "player":
                        w, h = x2 - x1, y2 - y1
                        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=1)
                        cvzone.putTextRect(img, f'{label} {conf}', (x1, y1 - 10), scale=0.7, thickness=1)

                cvzone.putTextRect(img, f'Unique Players: {len(unique_player_ids)}', (50, 50), scale=1, thickness=2)
                out.write(img)

            cap.release()
            out.release()
            print(f"Saved output to: {output_path}")

print(count_notprocessed)
print("✅ All videos processed.")

