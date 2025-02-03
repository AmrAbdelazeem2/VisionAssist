import os
import sys
import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO

# Get the directory of the script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define relative path for Depth-Anything
depth_anything_path = os.path.join(base_dir, "Depth-Anything-V2")

print("Base Directory:", base_dir)
print("Depth Anything Path:", depth_anything_path)
if depth_anything_path not in sys.path:
    sys.path.append(depth_anything_path)

from depth_anything_v2.dpt import DepthAnythingV2

# Load YOLOv8 model
model_yolo = YOLO('best.pt')
category_index = model_yolo.names

# Directories for output
temp_image_folder = os.path.join(base_dir, "temp_images")
processed_frames_folder = os.path.join(base_dir, "processed_frames")
os.makedirs(temp_image_folder, exist_ok=True)
os.makedirs(processed_frames_folder, exist_ok=True)

# Video input path
video_path = os.path.join(base_dir, 'input_videos', 'university_crosswalk.mp4')
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = fps
frame_count = 0
processed_frame_count = 0

### Initialize DepthAnythingV2 Model for Outdoor Metric Depth ###
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vits'
dataset = 'vkitti'
max_depth = 80

config = {**model_configs[encoder]}
depth_model = DepthAnythingV2(**config)
depth_model.max_depth = max_depth

checkpoint_path = os.path.join(depth_anything_path, 'checkpoints', f'depth_anything_v2_metric_{dataset}_{encoder}.pth')
depth_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
depth_model.eval()

def proximity_level(normalized_depth):
    if normalized_depth >= 0.8:
        return "Very Close"
    elif normalized_depth >= 0.5:
        return "Close"
    else:
        return "Far"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        processed_frame_count += 1
        print(f"Processing frame {processed_frame_count}...")

        # Infer metric depth map for the current frame
        depth_map = depth_model.infer_image(frame)

        # Normalize depth map
        min_depth = np.min(depth_map[depth_map > 0])
        max_depth_val = np.max(depth_map)

        results = model_yolo(frame)
        for result in results[0].boxes:
            box = result.xyxy[0].cpu().numpy()
            conf = float(result.conf.cpu().numpy().item())
            cls = int(result.cls.cpu().numpy().item())

            if conf > 0.60:
                startX, startY, endX, endY = box.astype(int)
                height, width = depth_map.shape
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(width, endX)
                endY = min(height, endY)

                # Crop the corresponding region from the depth map
                cropped_depth = depth_map[startY:endY, startX:endX]
                if cropped_depth.size > 0:
                    average_depth = np.mean(cropped_depth)
                    # Normalize depth
                    normalized_depth = (average_depth - min_depth) / (max_depth_val - min_depth + 1e-6)
                    normalized_depth = max(0, min(normalized_depth, 1))  # Clamp to [0, 1]
                    print(f'the normalized depth is: {normalized_depth}')
                    proximity = proximity_level(normalized_depth)
                else:
                    average_depth = 0
                    normalized_depth = 1  # Default to far if no valid depth
                    proximity = "Far"

                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                label = f"{category_index[cls]}: {normalized_depth:.2f} ({proximity})"
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                print(f"Detected {category_index[cls]} at [{startX}, {startY}, {endX}, {endY}] "
                      f"with confidence {conf:.2f}, normalized depth {normalized_depth:.2f}")

        frame_filename = f"frame_{processed_frame_count:04d}.jpg"
        processed_frame_path = os.path.join(processed_frames_folder, frame_filename)
        cv2.imwrite(processed_frame_path, frame)
        print(f"Annotated frame saved: {processed_frame_path}")

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("Processing complete.")