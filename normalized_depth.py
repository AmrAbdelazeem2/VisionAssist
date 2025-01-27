import os
import subprocess
import cv2
import time
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('best.pt')
category_index = model.names

# Paths for depth processing
base_dir = r"C:\Users\amrkh\OneDrive\Desktop\VisionAssist_SYSC4907"
depth_project_dir = os.path.join(base_dir, "Depth-Anything-V2")
temp_image_folder = os.path.join(base_dir, "temp_images")
depth_output_folder = os.path.join(base_dir, "depth_outputs")
processed_frames_folder = os.path.join(base_dir, "processed_frames")
model_encoder = "vits"

# Ensure necessary directories exist
os.makedirs(temp_image_folder, exist_ok=True)
os.makedirs(depth_output_folder, exist_ok=True)
os.makedirs(processed_frames_folder, exist_ok=True)

# Video input path
cap = cv2.VideoCapture(os.path.join(base_dir, 'input_videos', 'university_crosswalk.mp4'))
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
frame_interval = fps  # Process one frame per second
frame_count = 0
processed_frame_count = 0

# Function to run the Depth-Anything model
def process_depth(input_path, output_path):
    os.chdir(depth_project_dir)
    subprocess.run([
        "python", "run.py",
        "--encoder", model_encoder,
        "--img-path", input_path,
        "--outdir", output_path
    ], check=True)

# Function to wait for the depth output file
def wait_for_depth_output(file_path, timeout=10):
    elapsed_time = 0
    while elapsed_time < timeout:
        if os.path.exists(file_path):
            return True
        time.sleep(1)  # Wait for 1 second
        elapsed_time += 1
    return False

# Function to assign proximity levels based on normalized depth
def proximity_level(normalized_depth):
    if normalized_depth >= 0.8:  # Larger normalized depth means closer
        return "Very Close"
    elif normalized_depth >= 0.5:  # Mid-range depth means moderately close
        return "Close"
    else:  # Smaller normalized depth means far
        return "Far"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process every frame_interval frame
    if frame_count % frame_interval == 0:
        processed_frame_count += 1
        
        # Save the current frame for depth processing
        frame_filename = f"frame_{processed_frame_count:04d}.jpg"
        frame_path = os.path.join(temp_image_folder, frame_filename)
        cv2.imwrite(frame_path, frame)

        # Run the Depth-Anything model on the frame
        print(f"Processing depth for frame {processed_frame_count}...")
        process_depth(frame_path, depth_output_folder)

        # Define the depth output file path
        depth_image_path = os.path.join(depth_output_folder, frame_filename.replace(".jpg", ".png"))

        # Wait for the depth model to generate the output
        if wait_for_depth_output(depth_image_path, 60):
            depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                print(f"Error: Failed to load depth image for frame {processed_frame_count}.")
                continue
        else:
            print(f"Error: Depth output file not found for frame {processed_frame_count} after timeout.")
            continue

        # Compute percentile-based min and max depth
        min_depth = np.percentile(depth_image, 5)  # 5th percentile
        max_depth = np.percentile(depth_image, 95)  # 95th percentile
        print(f"Frame {processed_frame_count} - Min Depth: {min_depth}, Max Depth: {max_depth}")

        # Pass the frame to YOLO for object detection
        print(f"Processing YOLO object detection for frame {processed_frame_count}...")
        results = model(frame)
        for result in results[0].boxes:
            box = result.xyxy[0].cpu().numpy()  # bounding box
            conf = float(result.conf.cpu().numpy())    # Extract scalar value for confidence score
            cls = int(result.cls.cpu().numpy())        # class ID

            # If confidence level is 60 or more
            if conf > 0.60:
                startX, startY, endX, endY = box.astype(int)

                # Crop the corresponding region from the depth map
                cropped_depth = depth_image[startY:endY, startX:endX]
                if cropped_depth.size > 0:
                    average_depth = np.mean(cropped_depth)
                    # Normalize depth
                    normalized_depth = (average_depth - min_depth) / (max_depth - min_depth)
                    normalized_depth = max(0, min(normalized_depth, 1))  # Clamp to [0, 1]
                    print(f'the normalized depth is: {normalized_depth}')
                    proximity = proximity_level(normalized_depth)
                else:
                    average_depth = 0
                    normalized_depth = 1  # Default to far if no valid depth
                    proximity = "Far"

                # Draw bounding box around the object
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

                # Add label and depth information
                label = f"{category_index[cls]}: {proximity}"
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                print(f"Detected {category_index[cls]} at [{startX}, {startY}, {endX}, {endY}] "
                      f"with confidence {conf:.2f}, average depth {average_depth:.2f}, "
                      f"normalized depth {normalized_depth:.2f}, proximity: {proximity}")

        # Save the annotated frame
        processed_frame_path = os.path.join(processed_frames_folder, frame_filename)
        cv2.imwrite(processed_frame_path, frame)
        print(f"Annotated frame saved: {processed_frame_path}")

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("Processing complete.")