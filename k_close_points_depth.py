import os
import subprocess
import cv2
import time
import numpy as np
from ultralytics import YOLO

# Define the base directory dynamically
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define relative paths for depth processing
depth_project_dir = os.path.join(base_dir, "Depth-Anything-V2")
temp_image_folder = os.path.join(base_dir, "temp_original_images")
depth_output_folder = os.path.join(base_dir, "depth_outputs_images")
processed_frames_folder = os.path.join(base_dir, "processed_frames_images")
model_encoder = "vits"

# Ensure necessary directories exist
os.makedirs(temp_image_folder, exist_ok=True)
os.makedirs(depth_output_folder, exist_ok=True)
os.makedirs(processed_frames_folder, exist_ok=True)

# Load YOLOv8 model
model_yolo = YOLO(os.path.join(base_dir, 'best.pt'))
category_index = model_yolo.names

# Video input path
cap = cv2.VideoCapture('input_videos/university_crosswalk.mp4')
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
frame_interval = fps  # Process one frame per second
frame_count = 0
processed_frame_count = 0

# Function to run the Depth-Anything model
def process_depth(input_path, output_path):

    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    os.chdir(depth_project_dir)  # Ensure we're in the correct directory
    subprocess.run([
        "python", "run.py",
        "--encoder", model_encoder,
        "--img-path", input_path,
        "--outdir", output_path
    ], check=True)

# Function to wait for depth output (now we expect a .npy file)
def wait_for_depth_output(file_path, timeout=10):
    elapsed_time = 0
    while elapsed_time < timeout:
        if os.path.exists(file_path):
            return True
        time.sleep(1)
        elapsed_time += 1
    return False

# Function to assign proximity levels based on depth
def proximity_level(depth_value):
    if depth_value >= 7:
        return "Very Close"
    elif depth_value >= 4:
        return "Close"
    else:
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

        # ---------------------------------------------------------------------
        # UPDATED: We now look for the .npy file that run.py saves
        depth_npy_path = os.path.join(
            depth_output_folder,
            frame_filename.replace(".jpg", ".npy")
        )
        # ---------------------------------------------------------------------

        # Wait for the depth model to generate the .npy file
        if wait_for_depth_output(depth_npy_path, 60):
            # Load raw depth array (float32 or float64, depending on your model)
            raw_depth = np.load(depth_npy_path)  # <--- UPDATED

            if raw_depth.size == 0:
                print(f"Error: Failed to load valid depth array for frame {processed_frame_count}.")
                continue
            else:
                print("Raw Depth dtype:", raw_depth.dtype)
                print("Raw Depth shape:", raw_depth.shape)
                print("Raw Depth min/max:", raw_depth.min(), raw_depth.max())
        else:
            print(f"Error: Depth .npy file not found for frame {processed_frame_count} after timeout.")
            continue

        # Pass the frame to YOLO for object detection
        print(f"Processing YOLO object detection for frame {processed_frame_count}...")
        results = model_yolo(frame)
        for result in results[0].boxes:
            box = result.xyxy[0].cpu().numpy()  # bounding box
            conf = float(result.conf.cpu().numpy())    # confidence score
            cls = int(result.cls.cpu().numpy())        # class ID

            # If confidence level is 60% or more
            if conf > 0.60:
                startX, startY, endX, endY = box.astype(int)

                # "Shrink" the box a bit
                box_width = endX - startX
                box_height = endY - startY

                shrink_factor = 0.2  # shrink by 20% on each edge
                shrink_x = int(box_width * shrink_factor / 2)
                shrink_y = int(box_height * shrink_factor / 2)

                new_startX = startX + shrink_x
                new_endX   = endX   - shrink_x
                new_startY = startY + shrink_y
                new_endY   = endY   - shrink_y

                # Sample depth in [new_startY:new_endY, new_startX:new_endX]
                cropped_depth = raw_depth[new_startY:new_endY, new_startX:new_endX]  # <--- UPDATED

                if cropped_depth.size == 0:
                    print("No valid depth in bounding box.")
                else:
                    print("Crop min/max:", cropped_depth.min(), cropped_depth.max(), cropped_depth.shape)

                if cropped_depth.size > 0:
                    # Flatten and sort the depth values
                    depth_values = cropped_depth.flatten()
                    valid_values = depth_values[depth_values > 0] 
                    sorted_values = np.sort(valid_values)

                    # ignoring the top 5% as outliers, for example
                    cutoff_index = int(0.95 * len(sorted_values))
                    truncated_values = sorted_values[:cutoff_index]

                    k = 10  # Number of closest points to consider

                    if truncated_values.size >= k:
                        k_closest_points = truncated_values[-k:]  # last k
                        average_k_closest_depth = np.median(k_closest_points)
                    else:
                        average_k_closest_depth = np.median(truncated_values)
                                
                    # Determine proximity level
                    proximity = proximity_level(average_k_closest_depth)
                else:
                    average_k_closest_depth = 0
                    proximity = "Far"

                # Draw bounding box around the object
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

                # Add label and depth information
                label = f"{category_index[cls]}: {proximity} ({average_k_closest_depth:.2f})"
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                print(f"Detected {category_index[cls]} at [{startX}, {startY}, {endX}, {endY}] "
                      f"with confidence {conf:.2f}, k-closest depth {average_k_closest_depth:.2f}, proximity: {proximity}")

        # Save the annotated frame
        processed_frame_path = os.path.join(processed_frames_folder, frame_filename)
        cv2.imwrite(processed_frame_path, frame)
        print(f"Annotated frame saved: {processed_frame_path}")

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("Processing complete.")
