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
model_encoder = "vitl"

# Ensure necessary directories exist
os.makedirs(temp_image_folder, exist_ok=True)
os.makedirs(depth_output_folder, exist_ok=True)

# Video input path
cap = cv2.VideoCapture(os.path.join(base_dir, 'input_videos', 'trafficLightChange.mp4'))
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

# Process only the first frame
ret, frame = cap.read()
if ret:
    # Save the first frame as an image for depth processing
    frame_path = os.path.join(temp_image_folder, "frame.jpg")
    cv2.imwrite(frame_path, frame)

    # Run the Depth-Anything model on the frame
    print("Processing depth for the frame...")
    process_depth(frame_path, depth_output_folder)

    # Define the depth output file path
    depth_image_path = os.path.join(depth_output_folder, "frame.png")

    # Wait for the depth model to generate the output
    if wait_for_depth_output(depth_image_path, 60):
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        if depth_image is not None:
            print("Depth processing completed.")
        else:
            print("Error: Failed to load the depth image.")
    else:
        print(f"Error: Depth output file not found at {depth_image_path} after timeout.")
        cap.release()
        exit()

    # Pass the frame to YOLO for object detection
    print("Processing YOLO object detection...")
    results = model(frame)
    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()  # bounding box
        conf = float(result.conf.cpu().numpy())    # Extract scalar value for confidence score
        cls = int(result.cls.cpu().numpy())        # class ID

        if conf > 0.60:
            startX, startY, endX, endY = box.astype(int)

            # Crop the corresponding region from the depth map
            cropped_depth = depth_image[startY:endY, startX:endX]
            if cropped_depth.size > 0:
                average_depth = np.mean(cropped_depth)
            else:
                average_depth = 0  # Default depth if region is empty

            # Draw bounding box around the object
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

            # Add label and depth information
            label = f"{category_index[cls]}: {average_depth:.2f}"
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            print(f"Detected {category_index[cls]} at [{startX}, {startY}, {endX}, {endY}] "
                  f"with confidence {conf:.2f} and depth {average_depth:.2f}.")

    # Display the processed frame
    cv2.imshow("Processed Frame with Depth", frame)
    cv2.waitKey(0)  # Wait indefinitely for a key press
else:
    print("Error: Unable to read the first frame.")

cap.release()
cv2.destroyAllWindows()