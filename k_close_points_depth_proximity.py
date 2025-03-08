from ultralytics import YOLO
import cv2
import numpy as np
from transformers import pipeline
from PIL import Image

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
    # Convert BGR to RGB and then to PIL image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    
    # Run depth estimation
    depth_result = depth_estimator(pil_img)
    depth_tensor = depth_result["predicted_depth"]
    depth_array = depth_tensor.detach().cpu().numpy()
    
    # If the depth map size does not match the original frame, resize it
    if depth_array.shape[0] != frame.shape[0] or depth_array.shape[1] != frame.shape[1]:
        depth_array = cv2.resize(depth_array, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    return depth_array

def get_object_depth_k_median(depth_map, box_coords, k=50, iqr_multiplier=1.5):
    """
    Compute a representative depth value for a detected object by selecting
    the k highest depth values (i.e. the closest points, since higher means closer),
    and then taking the median of these points while ignoring outliers using IQR filtering.
    
    Parameters:
        depth_map (np.array): 2D array of depth values.
        box_coords (tuple): Bounding box as (x1, y1, x2, y2).
        k (int): Number of closest points to consider.
        iqr_multiplier (float): Multiplier for IQR to filter out outliers.
    
    Returns:
        float: Aggregated depth value or None if the region is empty.
    """
    x1, y1, x2, y2 = box_coords
    region = depth_map[y1:y2, x1:x2]
    if region.size == 0:
        return None

    # Flatten the region and sort depth values in descending order,
    # so that the highest (i.e. closest) values come first.
    flat_depths = region.flatten()
    sorted_depths = np.sort(flat_depths)[::-1]
    
    # Select the k highest depth values (the closest points)
    k = min(k, len(sorted_depths))
    k_points = sorted_depths[:k]
    
    # Compute the first and third quartiles and the IQR for these points
    q1 = np.percentile(k_points, 25)
    q3 = np.percentile(k_points, 75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    
    # Filter out outlier points
    filtered_points = k_points[(k_points >= lower_bound) & (k_points <= upper_bound)]
    
    # If no points remain after filtering, fallback to the median of the k closest points
    if filtered_points.size == 0:
        return np.median(k_points)
    
    return np.median(filtered_points)

def proximity_level_scene(object_depth, scene_depth_map):
    """
    Given an object's depth value and the scene's full depth map, assign a proximity level.
    
    Steps:
    1. Flatten the scene's depth map and filter out any non-positive values (if necessary).
    2. Compute the 80th and 60th percentiles.
    3. If object_depth >= 80th percentile: "Very Close"
       Else if object_depth >= 60th percentile: "Close"
       Else: "Far"
    """
    # Flatten the scene depth map
    all_depths = scene_depth_map.flatten()
    # Optionally, filter out non-positive or invalid depth values.
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
    
    # List to hold detected objects info: (box_coords, label, section, depth)
    objects_info = []
    
    # Run YOLO detection on the frame
    results = model(frame)
    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        conf = result.conf.cpu().numpy()
        cls = int(result.cls.cpu().numpy())
        
        if conf > 0.75:
            startX, startY, endX, endY = box.astype(int)
            x_center = (startX + endX) / 2
            
            # Determine section (left, center, right)
            if x_center < left_section:
                section = 'left'
            elif x_center > right_section:
                section = 'right'
            else:
                section = 'center'
            
            # Get object depth using the k-close median method
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
        
        # Draw bounding box and overlay text
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
        label_text = f'{label}: {section}, {depth_text}'
        cv2.putText(frame, label_text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        print(f"Detected {label} in {section} with {depth_text}")
    
    cv2.line(frame, (left_section, 0), (left_section, height), (0, 255, 0), 2)
    cv2.line(frame, (right_section, 0), (right_section, height), (0, 255, 0), 2)
    return frame

# Video capture initialization
cap = cv2.VideoCapture('input_videos/downtown2.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Process only a subset of frames (e.g., 4 frames per second)
process_fps = 4
frame_interval = fps // process_fps

# Video output writer
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
        
        # Show and write the processed frame
        cv2.imshow('YOLOv8 Detection with Depth', frame_with_detections)
        out.write(frame_with_detections)
    
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
