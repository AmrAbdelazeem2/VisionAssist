import cv2
import os
import time
from object_and_depth_estimation import model, category_index, get_depth_map, get_object_depth_k_median, proximity_level_scene
from audio_feedback import generate_audio_feedback

def process_frame(frame, queue):
    depth_map = get_depth_map(frame)
    detections = []

    results = model(frame)
    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()
        conf = result.conf.cpu().numpy()
        cls = int(result.cls.cpu().numpy().item())

        if conf > 0.75:
            startX, startY, endX, endY = box.astype(int)
            width = frame.shape[1]
            left_threshold = width // 3
            right_threshold = 2 * width // 3

            # Determine which section the detection falls in.
            left_overlap = max(0, min(endX, left_threshold) - startX)
            center_overlap = max(0, min(endX, right_threshold) - max(startX, left_threshold))
            right_overlap = max(0, endX - max(startX, right_threshold))
            overlaps = {'left': left_overlap, 'center': center_overlap, 'right': right_overlap}
            section = max(overlaps, key=overlaps.get)

            object_depth = get_object_depth_k_median(depth_map, (startX, startY, endX, endY))
            prox_level = proximity_level_scene(object_depth, depth_map) if object_depth is not None else "Far"

            detection = {
                "box": (startX, startY, endX, endY),
                "label": category_index[cls],
                "section": section,
                "depth": object_depth,
                "prox_level": prox_level
            }
            detections.append(detection)

            depth_text = f"Depth: {object_depth:.2f} ({prox_level})" if object_depth is not None else "Depth: N/A"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            label_text = f'{category_index[cls]}: {section}, {depth_text}'
            cv2.putText(frame, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            print(f"Detected {category_index[cls]} in {section} with {depth_text}")

    # Filter overlapping detections
    filtered_detections = []
    overlap_threshold = 0.3
    for i, det in enumerate(detections):
        discard = False
        for j, other_det in enumerate(detections):
            if i == j:
                continue
            if compute_iou(det["box"], other_det["box"]) > overlap_threshold:
                if det["depth"] is not None and other_det["depth"] is not None:
                    if det["depth"] < other_det["depth"]:
                        discard = True
                        break
        if not discard:
            filtered_detections.append(det)

    audio_objects = []
    for det in filtered_detections:
        if det["prox_level"] in ("Very Close", "Close"):
            audio_objects.append({
                "label": det["label"],
                "distance": det["prox_level"],
                "location": det["section"]
            })

    if audio_objects:
        queue.put({"objects": audio_objects})

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2b - x1b) * (y2b - y1b)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def detection_loop_folder(queue, folder_path):
    while True:
        # List all JPEG files in the buffer folder
        frame_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
        if not frame_files:
            time.sleep(0.1)  # Wait briefly if no file is available
            continue

        # Sort files by modification time (oldest first)
        frame_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))
        oldest_frame = os.path.join(folder_path, frame_files[0])
        frame = cv2.imread(oldest_frame)

        if frame is None:
            print(f"Could not read {oldest_frame}. Removing it.")
            os.remove(oldest_frame)
            continue

        process_frame(frame, queue)
        # Remove frame after processing
        os.remove(oldest_frame)

if __name__ == "__main__":
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser(description="Image detection with depth estimation and audio feedback from a folder.")
    parser.add_argument("--folder", type=str, default="./buffer", help="Path to the folder containing saved frames.")
    args = parser.parse_args()

    q = multiprocessing.Queue()
    detection_process = multiprocessing.Process(target=detection_loop_folder, args=(q, args.folder))
    detection_process.start()

    audio_process = multiprocessing.Process(target=generate_audio_feedback, args=(q,))
    audio_process.start()

    detection_process.join()
    audio_process.join()
