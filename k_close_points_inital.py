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
 
# Import DepthAnythingV2 from the correct module
from depth_anything_v2.dpt import DepthAnythingV2
 
# Load YOLOv8 model
model_yolo = YOLO('best.pt')
category_index = model_yolo.names
 
# Directories for output
temp_image_folder = os.path.join(base_dir, "temp_images")
processed_frames_folder = os.path.join(base_dir, "processed_frames")
os.makedirs(temp_image_folder, exist_ok=True)
os.makedirs(processed_frames_folder, exist_ok=True)
 
# Initialize DepthAnythingV2 Model
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
 
# Load model checkpoint
checkpoint_path = os.path.join(depth_anything_path, 'checkpoints', f'depth_anything_v2_metric_{dataset}_{encoder}.pth')
depth_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
depth_model.eval()
 
# Define proximity levels
def proximity_level(depth_value):
    if depth_value >= 180:
        return "Very Close"
    elif depth_value >= 150:
        return "Close"
    else:
        return "Far"
 
# Video input path
cap = cv2.VideoCapture(r'input_videos/university_crosswalk.mp4')
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()
 
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
frame_interval = fps  # Process one frame per second
frame_count = 0
processed_frame_count = 0
 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 
    # Process every frame_interval frame
    if frame_count % frame_interval == 0:
        processed_frame_count += 1
 
        # Ensure the image height and width are multiples of 14
        h, w, _ = frame.shape
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14
        resized_frame = cv2.resize(frame, (new_w, new_h))
 
        # Convert the resized frame to a tensor
        frame_tensor = torch.tensor(resized_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
 
        # Process depth
        with torch.no_grad():
            depth_output = depth_model(frame_tensor)
 
        depth_image = depth_output.squeeze().numpy()
 
        # Ensure depth image is valid
        if depth_image is None or depth_image.size == 0:
            print(f"Error: Failed to process depth for frame {processed_frame_count}.")
            continue
 
        print("Depth dtype:", depth_image.dtype)
        print("Depth shape:", depth_image.shape)
        print("Depth min/max:", depth_image.min(), depth_image.max())
 
        # Pass the frame to YOLO for object detection
        print(f"Processing YOLO object detection for frame {processed_frame_count}...")
        results = model_yolo(resized_frame)
 
        for result in results[0].boxes:
            box = result.xyxy[0].cpu().numpy()  # Bounding box
            conf = float(result.conf.cpu().numpy())  # Confidence score
            cls = int(result.cls.cpu().numpy())  # Class ID
 
            if conf > 0.60:  # Only process confident detections
                startX, startY, endX, endY = box.astype(int)
 
                # Crop the corresponding region from the depth map
                cropped_depth = depth_image[startY:endY, startX:endX]
 
                if cropped_depth.size == 0:
                    print("No valid depth in bounding box.")
                    continue
 
                print("Crop min/max:", cropped_depth.min(), cropped_depth.max(), cropped_depth.shape)
 
                # Flatten and sort depth values
                depth_values = cropped_depth.flatten()
                valid_values = depth_values[depth_values > 0]
                sorted_values = np.sort(valid_values)
 
                # Ignore top 10% outliers
                cutoff_index = int(0.95 * len(sorted_values))
                truncated_values = sorted_values[:cutoff_index]
 
                k = 10  # Number of closest points to consider
 
                if truncated_values.size >= k:
                    k_closest_points = truncated_values[-k:]  # Largest k values
                    average_k_closest_depth = np.median(k_closest_points)
                else:
                    average_k_closest_depth = np.median(truncated_values)
 
                # Determine proximity level
                proximity = proximity_level(average_k_closest_depth)
 
                # Draw bounding box around the object
                cv2.rectangle(resized_frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
 
                # Add label and depth information
                label = f"{category_index[cls]}: {proximity} ({average_k_closest_depth:.2f})"
                cv2.putText(resized_frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                print(f"Detected {category_index[cls]} at [{startX}, {startY}, {endX}, {endY}] "
                      f"with confidence {conf:.2f}, k-closest depth {average_k_closest_depth:.2f}, proximity: {proximity}")
 
        # Save the annotated frame
        frame_filename = f"frame_{processed_frame_count:04d}.jpg"
        processed_frame_path = os.path.join(processed_frames_folder, frame_filename)
        cv2.imwrite(processed_frame_path, resized_frame)
        print(f"Annotated frame saved: {processed_frame_path}")
 
    frame_count += 1
 
cap.release()
cv2.destroyAllWindows()
print("Processing complete.")
 
 