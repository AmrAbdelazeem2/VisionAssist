from ultralytics import YOLO
import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import multiprocessing
from AudioFeedbackSystem.AudioFeedback import generate_audio_feedback

# ------------------------- K_CLOSE_POINTS_DEPTH_PROXIMITY IMPLEMENTATION ------------------------- #

# Initialize YOLOv8 model
model = YOLO('best.pt')
category_index = model.names

# Initialize the depth estimation pipeline (relative depth by default)
depth_estimator = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

def get_depth_map(frame):
    """
    Convert a cv2 frame (BGR) to a PIL image, run the depth estimator,
    and return a NumPy array with depth values.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    depth_result = depth_estimator(pil_img)
    depth_tensor = depth_result["predicted_depth"]
    depth_array = depth_tensor.detach().cpu().numpy()
    if depth_array.shape[0] != frame.shape[0] or depth_array.shape[1] != frame.shape[1]:
        depth_array = cv2.resize(depth_array, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    return depth_array

def get_object_depth_k_median(depth_map, box_coords, k=50, iqr_multiplier=1.5):
    """
    Compute a representative depth value for a detected object by selecting
    the k highest depth values (i.e. the closest points, since higher means closer),
    and then taking the median of these points while ignoring outliers using IQR filtering.
    """
    x1, y1, x2, y2 = box_coords
    region = depth_map[y1:y2, x1:x2]
    if region.size == 0:
        return None
    flat_depths = region.flatten()
    sorted_depths = np.sort(flat_depths)[::-1]
    k = min(k, len(sorted_depths))
    k_points = sorted_depths[:k]
    q1 = np.percentile(k_points, 25)
    q3 = np.percentile(k_points, 75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    filtered_points = k_points[(k_points >= lower_bound) & (k_points <= upper_bound)]
    if filtered_points.size == 0:
        return np.median(k_points)
    return np.median(filtered_points)

def proximity_level_scene(object_depth, scene_depth_map):
    """
    Given an object's depth value and the scene's full depth map, assign a proximity level.
    """
    all_depths = scene_depth_map.flatten()
    valid_depths = all_depths[all_depths > 0]
    if valid_depths.size == 0:
        return "Unknown"
    q80 = np.percentile(valid_depths, 80)
    q60 = np.percentile(valid_depths, 60)
    if object_depth >= q80:
        return "Very Close"
    elif object_depth >= q60:
        return "Close"
    else:
        return "Far"

def detect_and_section(frame, depth_map):
    """
    Run object detection, determine the object's image section, and overlay
    both detection and depth info onto the frame.
    The proximity level is computed relative to the scene's overall depth distribution.
    """
    height, width, _ = frame.shape
    left_section = width // 3
    right_section = 2 * width // 3
    objects_info = []
    results = model(frame)
    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()
        conf = result.conf.cpu().numpy()
        cls = int(result.cls.cpu().numpy())
        if conf > 0.75:
            startX, startY, endX, endY = box.astype(int)
            x_center = (startX + endX) / 2
            if x_center < left_section:
                section = 'left'
            elif x_center > right_section:
                section = 'right'
            else:
                section = 'center'
            object_depth = get_object_depth_k_median(depth_map, (startX, startY, endX, endY))
            objects_info.append(((startX, startY, endX, endY), category_index[cls], section, object_depth))
    if not objects_info:
        return frame
    for (startX, startY, endX, endY), label, section, object_depth in objects_info:
        if object_depth is not None:
            prox_level = proximity_level_scene(object_depth, depth_map)
            depth_text = f"Depth: {object_depth:.2f} ({prox_level})"
        else:
            depth_text = "Depth: N/A"
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
        label_text = f'{label}: {section}, {depth_text}'
        cv2.putText(frame, label_text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        print(f"Detected {label} in {section} with {depth_text}")
    cv2.line(frame, (left_section, 0), (left_section, height), (0, 255, 0), 2)
    cv2.line(frame, (right_section, 0), (right_section, height), (0, 255, 0), 2)
    return frame

# --------------------- END OF K_CLOSE_POINTS_DEPTH_PROXIMITY IMPLEMENTATION --------------------- #

# --------------------- MULTIPROCESSING INTEGRATION (DO NOT CHANGE THE CORE IMPLEMENTATION) --------------------- #

def detection_loop(queue):
    """
    Video processing loop that:
      - Reads frames from the video.
      - Runs depth estimation and object detection.
      - Annotates the frame.
      - Sends detected object information via the queue for audio feedback.
    """
    cap = cv2.VideoCapture('input_videos/subway-test-cut.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    process_fps = 4
    frame_interval = fps // process_fps

    output_file = 'processed_videos/processed_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Get depth map for the current frame
            depth_map = get_depth_map(frame)
            
            # Detect objects and overlay depth and proximity information
            frame_with_detections = detect_and_section(frame, depth_map)
            
            # Extract objects info for audio feedback using the same detection logic
            left_section = width // 3
            right_section = 2 * width // 3
            objects_info = []
            results = model(frame)
            for result in results[0].boxes:
                box = result.xyxy[0].cpu().numpy()
                conf = result.conf.cpu().numpy()
                cls = int(result.cls.cpu().numpy())
                if conf > 0.75:
                    startX, startY, endX, endY = box.astype(int)
                    x_center = (startX + endX) / 2
                    if x_center < left_section:
                        location = 'left'
                    elif x_center > right_section:
                        location = 'right'
                    else:
                        location = 'middle'
                    object_depth = get_object_depth_k_median(depth_map, (startX, startY, endX, endY))
                    # Determine proximity using the scene's overall depth distribution
                    if object_depth is not None:
                        prox_level = proximity_level_scene(object_depth, depth_map)
                    else:
                        prox_level = "Unknown"
                    objects_info.append({
                        "label": category_index[cls],
                        "location": location,
                        "distance": prox_level
                    })
            if objects_info:
                queue.put({"objects": objects_info})
            
            cv2.imshow('YOLOv8 Detection with Depth', frame_with_detections)
            out.write(frame_with_detections)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Detection process complete.")

if __name__ == "__main__":
    # Create a multiprocessing queue for communication between processes.
    queue = multiprocessing.Queue()

    # Start the detection process (video processing).
    detection_process = multiprocessing.Process(target=detection_loop, args=(queue,))
    detection_process.start()

    # Start the audio feedback process.
    audio_process = multiprocessing.Process(target=generate_audio_feedback, args=(queue,))
    audio_process.start()

    detection_process.join()
    audio_process.join()
