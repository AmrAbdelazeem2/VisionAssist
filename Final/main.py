import cv2
import multiprocessing
import argparse
from object_and_depth_estimation import model, category_index, get_depth_map, get_object_depth_k_median, proximity_level_scene
from audio_feedback import generate_audio_feedback

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

def detection_loop(queue, video_path, process_fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Divide the frame into three equal sections.
    left_threshold = width // 3
    right_threshold = 2 * width // 3

    frame_interval = max(1, fps // process_fps)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            depth_map = get_depth_map(frame)
            detections = []

            results = model(frame)
            for result in results[0].boxes:
                box = result.xyxy[0].cpu().numpy()
                conf = result.conf.cpu().numpy()
                # Fix deprecation warning by extracting the scalar value.
                cls = int(result.cls.cpu().numpy().item())

                if conf > 0.75:
                    startX, startY, endX, endY = box.astype(int)
                    # Calculate horizontal overlaps with equal sections.
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

                    # Draw bounding box and label.
                    depth_text = f"Depth: {object_depth:.2f} ({prox_level})" if object_depth is not None else "Depth: N/A"
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                    label_text = f'{category_index[cls]}: {section}, {depth_text}'
                    cv2.putText(frame, label_text, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    print(f"Detected {category_index[cls]} in {section} with {depth_text}")

            # Filter overlapping detections; for IoU > threshold, keep the detection with the closer (higher) depth.
            filtered_detections = []
            overlap_threshold = 0.3
            for i, det in enumerate(detections):
                discard = False
                for j, other_det in enumerate(detections):
                    if i == j:
                        continue
                    #TODO: improve the iou computation to consider the depth information more accurately.
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

            cv2.imshow('Detection with Audio Feedback', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

# To run the code:
# python main.py --video "../input_videos/university_crosswalk.mp4" --fps 2
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object detection with depth estimation and audio feedback.")
    parser.add_argument("--video", type=str, default="../input_videos/university_crosswalk.mp4", help="Path to input video file.")
    parser.add_argument("--fps", type=int, default=2, help="Number of frames per second to process.")
    args = parser.parse_args()

    q = multiprocessing.Queue()
    detection_process = multiprocessing.Process(target=detection_loop, args=(q, args.video, args.fps))
    detection_process.start()
    audio_process = multiprocessing.Process(target=generate_audio_feedback, args=(q,))
    audio_process.start()

    detection_process.join()
    audio_process.join()
