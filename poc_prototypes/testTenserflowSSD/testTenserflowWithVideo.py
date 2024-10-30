import cv2
import numpy as np
import tensorflow as tf


MODEL_DIR = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model'

# Load the pre-trained TensorFlow Object Detection model
detect_fn = tf.saved_model.load(MODEL_DIR)

# COCO dataset with display_name
def load_labels(label_file):
    labels = {}
    with open(label_file, 'r') as f:
        current_id = None
        display_name = None
        for line in f:
            if "id" in line:
                current_id = int(line.split(":")[1].strip())
            elif "display_name" in line:
                display_name = line.split(":")[1].strip().replace('"', '') 
                labels[current_id] = display_name
    return labels


LABEL_MAP_PATH = 'mscoco_label_map.pbtxt'


category_index = load_labels(LABEL_MAP_PATH)

cap = cv2.VideoCapture('university.mp4')

# Get the original video's width, height, and FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create VideoWriter object to save the processed video
output_file = 'processed_output.mp4' 
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Function to perform object detection and section mapping
def detect_and_section(frame):
    # Get frame dimensions
    height, width, _ = frame.shape

    # Split frame into sections (left, center, right)
    left_section = width // 3
    right_section = 2 * width // 3

    # Convert frame to tensor and make predictions
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    # Process the results
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    boxes = detections['detection_boxes']
    classes = detections['detection_classes'].astype(np.int64)
    scores = detections['detection_scores']

    # Loop through all detected objects
    for i in range(num_detections):
        # check confidence threshold for the detected objects
        if scores[i] > 0.5:
            # get the dimensions of the bounding box to know in what section it is located
            ymin, xmin, ymax, xmax = boxes[i]
            x_center = (xmin + xmax) / 2 * width 

            if x_center < left_section:
                section = 'left'
            elif x_center > right_section:
                section = 'right'
            else:
                section = 'center'

            # Draw bounding box around the object
            (startX, startY, endX, endY) = (int(xmin * width), int(ymin * height),
                                            int(xmax * width), int(ymax * height))
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

            # Get the class label for the object detected
            label = category_index.get(classes[i], 'Unknown')

            # Put label and section info on the frame
            label_text = f'{label}: {section}'
            cv2.putText(frame, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # print feedback
            print(f"Detected {label} in the {section} section")

    # Draw section lines on the frame for visualizing the sections
    cv2.line(frame, (left_section, 0), (left_section, height), (0, 255, 0), 2)
    cv2.line(frame, (right_section, 0), (right_section, height), (0, 255, 0), 2)

    return frame

# capture video and process the frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    frame_with_detections = detect_and_section(frame)

    # Write the processed frame to the video file
    out.write(frame_with_detections)

# Release the video capture and writer
cap.release()
out.release()