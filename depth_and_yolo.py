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

def get_object_depth(depth_map, box_coords, method="median"):
    """
    Given a depth map and bounding box coordinates, compute a representative depth.
    
    Parameters:
        depth_map (np.array): 2D array of depth values.
        box_coords (tuple): Bounding box as (x1, y1, x2, y2).
        method (str): "median" or "mean" to aggregate the depth values.
    
    Returns:
        float: Aggregated depth value.
    """
    x1, y1, x2, y2 = box_coords
    region = depth_map[y1:y2, x1:x2]
    if region.size == 0:
        return None
    if method == "median":
        return np.median(region)
    elif method == "mean":
        return np.mean(region)
    else:
        raise ValueError("Aggregation method must be 'median' or 'mean'.")

def detect_and_section(frame, depth_map):
    """
    Run object detection, determine the object's image section, and overlay
    both detection and depth info onto the frame.
    """
    height, width, _ = frame.shape
    left_section = width // 3
    right_section = 2 * width // 3
    
    # Run YOLO detection on the frame
    results = model(frame)
    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()  # Bounding box: [x1, y1, x2, y2]
        conf = result.conf.cpu().numpy()      # Confidence score
        cls = int(result.cls.cpu().numpy())   # Class ID
        
        if conf > 0.75:
            startX, startY, endX, endY = box.astype(int)
            x_center = (startX + endX) / 2
            
            # Determine the section (left, center, right)
            if x_center < left_section:
                section = 'left'
            elif x_center > right_section:
                section = 'right'
            else:
                section = 'center'
            
            # Extract depth info for the detected object
            object_depth = get_object_depth(depth_map, (startX, startY, endX, endY))
            depth_text = f"Depth: {object_depth:.2f}" if object_depth is not None else "Depth: N/A"
            
            # Draw bounding box and text
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            label = category_index[cls]
            label_text = f'{label}: {section}, {depth_text}'
            cv2.putText(frame, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Debug print
            print(f"Detected {label} in {section} section with {depth_text}")
    
    # Draw dividing lines for the image sections
    cv2.line(frame, (left_section, 0), (left_section, height), (0, 255, 0), 2)
    cv2.line(frame, (right_section, 0), (right_section, height), (0, 255, 0), 2)
    return frame

# Video capture initialization
# university_crosswalk
cap = cv2.VideoCapture('input_videos/university_walking.mp4')
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
        
        # Detect objects and overlay depth information
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
