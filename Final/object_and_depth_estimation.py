from ultralytics import YOLO
import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
from sklearn.cluster import KMeans

# Initialize YOLOv8 model and depth estimator.
model = YOLO('best.pt')
category_index = model.names
depth_estimator = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

def get_depth_map(frame):
    """
    Convert a cv2 BGR frame to a PIL image, run the depth estimator,
    and return a NumPy array with depth values.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    depth_result = depth_estimator(pil_img)
    depth_tensor = depth_result["predicted_depth"]
    depth_array = depth_tensor.detach().cpu().numpy()
    if depth_array.shape[:2] != frame.shape[:2]:
        depth_array = cv2.resize(depth_array, (frame.shape[1], frame.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
    return depth_array

def get_object_depth_k_median(depth_map, box_coords, k=50, iqr_multiplier=1.5):
    """
    Compute a representative depth value for a detected object by taking the k
    highest depth values (i.e. the closest points) and computing their median
    after filtering out outliers using IQR.
    """
    x1, y1, x2, y2 = box_coords
    region = depth_map[y1:y2, x1:x2]
    if region.size == 0:
        return None

    flat_depths = region.flatten()
    sorted_depths = np.sort(flat_depths)[::-1]
    k = min(k, len(sorted_depths))
    k_points = sorted_depths[:k]
    
    q1, q3 = np.percentile(k_points, 25), np.percentile(k_points, 75)
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
    filtered_points = k_points[(k_points >= lower_bound) & (k_points <= upper_bound)]
    
    return np.median(filtered_points) if filtered_points.size > 0 else np.median(k_points)

def proximity_level_scene(object_depth, scene_depth_map):
    """
    Determine the proximity level ("Very Close", "Close", or "Far") by clustering
    the overall scene's depth values using k-means and mapping the object's depth.
    """
    valid_depths = scene_depth_map.flatten()
    valid_depths = valid_depths[valid_depths > 0]
    if valid_depths.size == 0:
        return "Unknown"
    
    valid_depths = valid_depths.reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(valid_depths)
    centers = kmeans.cluster_centers_.flatten()
    sorted_indices = centers.argsort()[::-1]
    cluster_label_map = {
        sorted_indices[0]: "Very Close",
        sorted_indices[1]: "Close",
        sorted_indices[2]: "Far"
    }
    distances = np.abs(centers - object_depth)
    object_cluster = distances.argmin()
    return cluster_label_map[object_cluster]

def detect_and_section(frame, depth_map):
    """
    Run object detection on the frame, estimate each object's depth, and overlay
    bounding boxes and depth information. The frame is divided into three equal sections.
    If an object's bounding box overlaps multiple sections, the section that covers the largest
    portion of the object horizontally is assigned.
    """
    height, width, _ = frame.shape
    # Divide the frame into three equal sections.
    left_threshold = width // 3
    right_threshold = 2 * width // 3

    def get_overlap(start, end, sec_start, sec_end):
        return max(0, min(end, sec_end) - max(start, sec_start))

    objects_info = []
    
    results = model(frame)
    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()
        conf = result.conf.cpu().numpy()
        cls = int(result.cls.cpu().numpy().item())
        
        if conf > 0.75:
            startX, startY, endX, endY = box.astype(int)
            # Compute horizontal overlaps with each section.
            left_overlap = get_overlap(startX, endX, 0, left_threshold)
            center_overlap = get_overlap(startX, endX, left_threshold, right_threshold)
            right_overlap = get_overlap(startX, endX, right_threshold, width)
            
            overlaps = {'left': left_overlap, 'center': center_overlap, 'right': right_overlap}
            section = max(overlaps, key=overlaps.get)
            
            object_depth = get_object_depth_k_median(depth_map, (startX, startY, endX, endY))
            objects_info.append(((startX, startY, endX, endY), category_index[cls], section, object_depth))
    
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
    
    # Draw vertical lines to visualize section boundaries.
    cv2.line(frame, (left_threshold, 0), (left_threshold, height), (0, 255, 0), 2)
    cv2.line(frame, (right_threshold, 0), (right_threshold, height), (0, 255, 0), 2)
    
    return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(r'../input_videos/university_crosswalk.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    process_fps = 2
    frame_interval = fps // process_fps
    output_file = '../processed_videos/processed_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            depth_map = get_depth_map(frame)
            processed_frame = detect_and_section(frame, depth_map)
            cv2.imshow('YOLOv8 Detection with Depth', processed_frame)
            out.write(processed_frame)
        
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
