import os
import subprocess
import cv2
import time
import numpy as np
import concurrent.futures
from ultralytics import YOLO

# Load the YOLOv8 model
model_yolo = YOLO('best.pt')
category_index = model_yolo.names

# Paths for depth processing
base_dir = "/Users/samehgawish/Desktop/Capstone/ProjectRepo/VisionAssist_SYSC4907"
depth_project_dir = os.path.join(base_dir, "Depth-Anything-V2")
temp_image_folder = os.path.join(base_dir, "temp_original_images")
depth_output_folder = os.path.join(base_dir, "depth_outputs_images")
processed_frames_folder = os.path.join(base_dir, "processed_frames_images")

# Ensure necessary directories exist
os.makedirs(temp_image_folder, exist_ok=True)
os.makedirs(depth_output_folder, exist_ok=True)
os.makedirs(processed_frames_folder, exist_ok=True)

model_encoder = "vits"

# Video input path
cap = cv2.VideoCapture('input_videos/VIDEO-2025-02-23-19-32-23.mp4')
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
frame_interval = fps  # Process one frame per second
frame_count = 0
processed_frame_count = 0

# Global buffer to store the last few depth images for temporal smoothing
depth_history = []
history_buffer_size = 3  # You can adjust this based on your requirements

# Function to run the Depth-Anything model
def process_depth(input_path, output_path):
    run_script_path = os.path.join(depth_project_dir, "run.py")
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    subprocess.run([
        "python",
        run_script_path,
        "--encoder", model_encoder,
        "--img-path", input_path,
        "--outdir", output_path
    ], check=True, env=env)

# Function to wait for the depth output file
def wait_for_depth_output(file_path, timeout=10):
    elapsed_time = 0
    while elapsed_time < timeout:
        if os.path.exists(file_path):
            return True
        time.sleep(1)  # Wait for 1 second
        elapsed_time += 1
    return False

# Wrapper function to process depth for a frame (to be run in a thread)
def process_depth_frame(frame, frame_filename):
    # Save the frame to disk for depth processing
    frame_path = os.path.join(temp_image_folder, frame_filename)
    cv2.imwrite(frame_path, frame)
    
    # Run the depth model on the saved frame
    process_depth(frame_path, depth_output_folder)
    
    # Define the expected depth output path (assuming .png output)
    depth_image_path = os.path.join(depth_output_folder, frame_filename.replace(".jpg", ".png"))
    
    # Wait for the depth model to generate the output
    if wait_for_depth_output(depth_image_path, 60):
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
        if depth_image is None:
            print(f"Error: Failed to load depth image for {frame_filename}.")
        return depth_image
    else:
        print(f"Error: Depth output file not found for {frame_filename} after timeout.")
        return None

# Function to aggregate depth values across the history for a given bounding box
def aggregate_depth(bbox, depth_images):
    startX, startY, endX, endY = bbox
    all_depth_values = []
    for depth_img in depth_images:
        # Ensure bounding box is within image dimensions
        h, w = depth_img.shape
        x1 = max(0, startX)
        y1 = max(0, startY)
        x2 = min(w, endX)
        y2 = min(h, endY)
        cropped = depth_img[y1:y2, x1:x2]
        if cropped.size > 0:
            valid_values = cropped[cropped > 0]
            if valid_values.size > 0:
                all_depth_values.extend(valid_values.flatten().tolist())
    if len(all_depth_values) == 0:
        return None
    sorted_values = np.sort(np.array(all_depth_values))
    # Remove extreme outliers (using 95% cutoff)
    cutoff_index = int(0.95 * len(sorted_values))
    truncated_values = sorted_values[:cutoff_index]
    k = 10  # Number of closest points to consider
    if truncated_values.size >= k:
        k_closest_points = truncated_values[-k:]
        return np.median(k_closest_points)
    else:
        return np.median(truncated_values)

# Function to determine proximity level based on aggregated depth
def proximity_level(depth_value):
    if depth_value is None:
        return "Unknown"
    if depth_value >= 2000:
        return "Very Close"
    elif depth_value >= 150:
        return "Close"
    else:
        return "Far"

# Use a ThreadPoolExecutor for asynchronous depth processing
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every frame_interval frame
        if frame_count % frame_interval == 0:
            processed_frame_count += 1
            frame_filename = f"frame_{processed_frame_count:04d}.jpg"
            
            print(f"Submitting depth processing for frame {processed_frame_count}...")
            future = executor.submit(process_depth_frame, frame, frame_filename)
            depth_image = future.result()  # Wait for depth result

            # If a valid depth image is returned, add it to the history buffer
            if depth_image is not None:
                depth_history.append(depth_image)
                if len(depth_history) > history_buffer_size:
                    depth_history.pop(0)

            # Process YOLO object detection on the current frame
            print(f"Processing YOLO object detection for frame {processed_frame_count}...")
            results = model_yolo(frame)
            for result in results[0].boxes:
                box = result.xyxy[0].cpu().numpy()  # Bounding box coordinates
                conf = float(result.conf.cpu().numpy())
                cls = int(result.cls.cpu().numpy())
                
                # Consider detections with high confidence only
                if conf > 0.60:
                    startX, startY, endX, endY = box.astype(int)
                    bbox = (startX, startY, endX, endY)

                    # Aggregate depth from the history buffer for better accuracy
                    aggregated_depth = aggregate_depth(bbox, depth_history)
                    if aggregated_depth is None:
                        aggregated_depth = 0  # Default if no valid data is found
                    proximity = proximity_level(aggregated_depth)
                    
                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                    label = f"{category_index[cls]}: {proximity} ({aggregated_depth:.2f})"
                    cv2.putText(frame, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    print(f"Detected {category_index[cls]} at [{startX}, {startY}, {endX}, {endY}] "
                          f"with confidence {conf:.2f}, aggregated depth {aggregated_depth:.2f}, proximity: {proximity}")

            # Save the annotated frame
            processed_frame_path = os.path.join(processed_frames_folder, frame_filename)
            cv2.imwrite(processed_frame_path, frame)
            print(f"Annotated frame saved: {processed_frame_path}")

        frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("Processing complete.")
