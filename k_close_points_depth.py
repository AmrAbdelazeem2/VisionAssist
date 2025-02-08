import os
import subprocess
import cv2
import time
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model_yolo = YOLO('best.pt')
category_index = model_yolo.names

# Paths for depth processing
base_dir = "/Users/samehgawish/Desktop/Capstone/ProjectRepo/VisionAssist_SYSC4907"
depth_project_dir = os.path.join(base_dir, "Depth-Anything-V2/Depth-Anything-V2")
temp_image_folder = os.path.join(base_dir, "temp_original_images")
depth_output_folder = os.path.join(base_dir, "depth_outputs_images")
processed_frames_folder = os.path.join(base_dir, "processed_frames_images")
model_encoder = "vits"

# Ensure necessary directories exist
os.makedirs(temp_image_folder, exist_ok=True)
os.makedirs(depth_output_folder, exist_ok=True)
os.makedirs(processed_frames_folder, exist_ok=True)

# Video input path
cap = cv2.VideoCapture('university_crosswalk.mp4')
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
frame_interval = fps  # Process one frame per second
frame_count = 0
processed_frame_count = 0

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

# Function to assign proximity levels based on depth
def proximity_level(depth_value):
    if depth_value >= 2000:
        return "Very Close"
    elif depth_value >= 150:
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

        # Define the depth output file path
        depth_image_path = os.path.join(depth_output_folder, frame_filename.replace(".jpg", ".png"))

        # Wait for the depth model to generate the output
        if wait_for_depth_output(depth_image_path, 60):
            depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)

            if depth_image is None:
                print(f"Error: Failed to load depth image for frame {processed_frame_count}.")
                continue
            else:
                print("Depth dtype:", depth_image.dtype)
                print("Depth shape:", depth_image.shape)
                print("Depth min/max:", depth_image.min(), depth_image.max())
                
        else:
            print(f"Error: Depth output file not found for frame {processed_frame_count} after timeout.")
            continue

        # Pass the frame to YOLO for object detection
        print(f"Processing YOLO object detection for frame {processed_frame_count}...")
        results = model_yolo(frame)
        for result in results[0].boxes:
            box = result.xyxy[0].cpu().numpy()  # bounding box
            conf = float(result.conf.cpu().numpy())    # Extract scalar value for confidence score
            cls = int(result.cls.cpu().numpy())        # class ID

            # If confidence level is 60 or more
            if conf > 0.60:
                startX, startY, endX, endY = box.astype(int)

                # Crop the corresponding region from the depth map
                cropped_depth = depth_image[startY:endY, startX:endX]


                if cropped_depth.size == 0:
                    print("No valid depth in bounding box.")
                else:
                    print("Crop min/max:", cropped_depth.min(), cropped_depth.max(), cropped_depth.shape)

                if cropped_depth.size > 0:
                    # Flatten and sort the depth values to find the k closest points
                    depth_values = cropped_depth.flatten()
                    valid_values = depth_values[depth_values > 0] 
                    sorted_values = np.sort(valid_values)

                    # ignoring the 10% max points as outliers
                    cutoff_index = int(0.95 * len(sorted_values))  # 90% index
                    truncated_values = sorted_values[:cutoff_index]


                    k = 10  # Number of closest points to consider

                    if truncated_values.size >= k:
                        k_closest_points = truncated_values[-k:]  # last k = largest k
                        average_k_closest_depth = np.median(k_closest_points)
                    else:
                        # If not enough data for k, just take the median of what's left
                        average_k_closest_depth = np.median(truncated_values)
                                
                    # Determine proximity level
                    proximity = proximity_level(average_k_closest_depth)
                else:
                    average_k_closest_depth = 0  # Default depth if no valid region
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