import os
import subprocess
import cv2
import time
import numpy as np
import multiprocessing
from ultralytics import YOLO
from AudioFeedbackSystem.AudioFeedback import generate_audio_feedback

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

# Define video input path
video_path = os.path.join(base_dir, 'input_videos', 'university_crosswalk.mp4')
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
frame_interval = fps  # Process one frame per second
frame_count = 0
processed_frame_count = 0

# Function to run Depth-Anything model
def process_depth(input_path, output_path):
    os.chdir(depth_project_dir)  # Ensure we're in the correct directory
    subprocess.run([
        "python", "run.py",
        "--encoder", model_encoder,
        "--img-path", input_path,
        "--outdir", output_path
    ], check=True)

# Function to wait for depth output
def wait_for_depth_output(file_path, timeout=10):
    elapsed_time = 0
    while elapsed_time < timeout:
        if os.path.exists(file_path):
            return True
        time.sleep(1)
        elapsed_time += 1
    return False

# Function to assign distance labels based on depth
def assign_distance_label(depth_value):
    if depth_value >= 2000:
        return "immediate"
    elif depth_value >= 150:
        return "close"
    elif depth_value >= 50:
        return "moderate"
    elif depth_value >= 10:
        return "far"
    else:
        return "very far"

# Function to determine object location (left, middle, right)
def get_location(startX, endX, frame_width):
    mid_x = (startX + endX) / 2
    if mid_x < frame_width * 0.33:
        return "left"
    elif mid_x > frame_width * 0.66:
        return "right"
    else:
        return "middle"

# Function to process frames and send detected objects to the queue
def detect_objects(queue):
    global frame_count, processed_frame_count

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            processed_frame_count += 1

            frame_filename = f"frame_{processed_frame_count:04d}.jpg"
            frame_path = os.path.join(temp_image_folder, frame_filename)
            cv2.imwrite(frame_path, frame)

            # Run the Depth-Anything model
            print(f"Processing depth for frame {processed_frame_count}...")
            process_depth(frame_path, depth_output_folder)

            # Define depth output file path
            depth_image_path = os.path.join(depth_output_folder, frame_filename.replace(".jpg", ".png"))

            # Wait for depth processing
            if wait_for_depth_output(depth_image_path, 60):
                depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
                if depth_image is None:
                    print(f"Error: Failed to load depth image for frame {processed_frame_count}.")
                    continue
            else:
                print(f"Error: Depth output file not found for frame {processed_frame_count} after timeout.")
                continue

            # Get frame width
            frame_width = frame.shape[1]

            # Run YOLO object detection
            print(f"Processing YOLO object detection for frame {processed_frame_count}...")
            results = model_yolo(frame)
            detected_objects = []

            for result in results[0].boxes:
                box = result.xyxy[0].cpu().numpy()
                conf = float(result.conf.cpu().numpy())
                cls = int(result.cls.cpu().numpy())

                if conf > 0.60:
                    startX, startY, endX, endY = box.astype(int)

                    # Determine object location
                    location = get_location(startX, endX, frame_width)

                    # Extract depth region
                    cropped_depth = depth_image[startY:endY, startX:endX]

                    if cropped_depth.size > 0:
                        depth_values = cropped_depth.flatten()
                        valid_values = depth_values[depth_values > 0]
                        sorted_values = np.sort(valid_values)

                        cutoff_index = int(0.95 * len(sorted_values))
                        truncated_values = sorted_values[:cutoff_index]

                        k = 10
                        if truncated_values.size >= k:
                            k_closest_points = truncated_values[-k:]
                            average_k_closest_depth = np.median(k_closest_points)
                        else:
                            average_k_closest_depth = np.median(truncated_values)

                        distance_label = assign_distance_label(average_k_closest_depth)

                        detected_objects.append({
                            "label": category_index[cls],
                            "distance": distance_label,
                            "location": location
                        })

                        # Draw bounding box & label
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                        label_text = f"{category_index[cls]} ({distance_label}) - {location}"
                        cv2.putText(frame, label_text, (startX, startY - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        print(f"Detected {category_index[cls]} at {location} with confidence {conf:.2f}, "
                              f"depth {average_k_closest_depth:.2f}, distance: {distance_label}")

            # Save the annotated frame even if no objects are detected
            processed_frame_path = os.path.join(processed_frames_folder, frame_filename)
            cv2.imwrite(processed_frame_path, frame)
            print(f"Annotated frame saved: {processed_frame_path}")

            # Send detected objects to the queue for audio feedback
            if detected_objects:
                queue.put({"objects": detected_objects})

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Detection process complete.")

if __name__ == "__main__":
    queue = multiprocessing.Queue()

    # Start Object Detection Process
    detection_process = multiprocessing.Process(target=detect_objects, args=(queue,))
    detection_process.start()

    # Start Audio Feedback Process
    audio_process = multiprocessing.Process(target=generate_audio_feedback, args=(queue,))
    audio_process.start()

    detection_process.join()
    audio_process.join()