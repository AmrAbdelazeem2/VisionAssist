import cv2
import numpy as np
import tensorflow as tf

# Adjust the path to your saved model
MODEL_DIR = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model'

# Load the pre-trained TensorFlow Object Detection model
detect_fn = tf.saved_model.load(MODEL_DIR)

# Helper function to load labels (e.g., COCO dataset)
def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels = {}
        for line in f.readlines():
            parts = line.strip().split(':')
            if len(parts) == 2:
                labels[int(parts[0])] = parts[1].strip()
        return labels

# Adjust this path to point to your label map file
LABEL_MAP_PATH = 'mscoco_label_map.pbtxt'

# Load label map (ensure this file is available with your model)
category_index = load_labels(LABEL_MAP_PATH)

# Video capture (use 0 for default camera, or specify path to video)
cap = cv2.VideoCapture(0)

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
        if scores[i] > 0.5:  # Confidence threshold
            ymin, xmin, ymax, xmax = boxes[i]
            x_center = (xmin + xmax) / 2 * width  # Get the x-center of the bounding box

            # Determine which section the object is in
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

            # Output feedback based on section and detected object
            print(f"Detected {label} in the {section} section")

    # Draw section lines on the frame for visualization
    cv2.line(frame, (left_section, 0), (left_section, height), (0, 255, 0), 2)
    cv2.line(frame, (right_section, 0), (right_section, height), (0, 255, 0), 2)

    return frame

# Main loop for capturing video and processing frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects and map to sections
    frame_with_detections = detect_and_section(frame)

    # Display the processed frame (for debugging purposes)
    cv2.imshow('Obstacle Detection and Sectioning', frame_with_detections)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()



# for audio feedback
# import pyttsx3
# engine = pyttsx3.init()
# engine.say(f"Detected {label} in the {section} section")
# engine.runAndWait()