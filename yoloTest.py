from ultralytics import YOLO
import cv2

# Load the YOLOv8 model (Change to yolov8m.pt for the medium version)
model = YOLO('yolov8s.pt')
category_index = model.names

# NOTE: change the path below for your corresponding video path
cap = cv2.VideoCapture('input_videos/videoplayback.mp4')
cap = cv2.VideoCapture('../../university_crosswalk.mp4')

# Get the original video's coordinates: width, height, and FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# video output writer
output_file = 'processed_videos/processed_output.mp4' #update this path if you want a specific output video name 
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))


"""Detect the objects in the frame using the chosen model,
    and link them to some section in the image (e.g. left_section)
    
    args: 
    frame (MatLike) : video frame
"""
def detect_and_section(frame):
    height, width, _ = frame.shape
    left_section = width // 3
    right_section = 2 * width // 3
    
    # Perform object detection on the frame
    results = model(frame)

    # iterate the detected objects in the frame
    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()  # bounding box
        conf = result.conf.cpu().numpy()    # confidence score
        cls = int(result.cls.cpu().numpy()) # class ID

        
        if conf > 0.5:
            startX, startY, endX, endY = box.astype(int)
            x_center = (startX + endX) / 2

            # Determine the section (left, center, right)
            if x_center < left_section:
                section = 'left'
            elif x_center > right_section:
                section = 'right'
            else:
                section = 'center'

            # Draw bounding box around the object
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            # Get the object label
            label = category_index[cls]
            # Put label and section info on the frame
            label_text = f'{label}: {section}'
            cv2.putText(frame, label_text, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Print feedback (NOTE: just for debug purposes and to be deleted)
            print(f"Detected {label} in the {section} section")

    
    cv2.line(frame, (left_section, 0), (left_section, height), (0, 255, 0), 2)
    cv2.line(frame, (right_section, 0), (right_section, height), (0, 255, 0), 2)
    return frame


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_with_detections = detect_and_section(frame)

    # show the process frame (NOTE: for debug and visualization purpose only )
    cv2.imshow('YOLOv8 Real-time Detection', frame_with_detections)
    
    out.write(frame_with_detections)

    # quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()