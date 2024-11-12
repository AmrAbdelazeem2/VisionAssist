import torch
import cv2

# Load the YOLOR model (ensure yolor_p6.pt is in the appropriate directory)
model = torch.hub.load('WongKinYiu/yolor', 'yolor_p6', pretrained=True)
model.eval()  # Set the model to evaluation mode

# Get the class names
category_index = model.names

# Change the path to your video file as needed
cap = cv2.VideoCapture('input_videos/university_crosswalk.mp4')

# Get the original video's properties: width, height, and FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set up video writer
output_file = 'processed_videos/processed_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))


def detect_and_section(frame):
    height, width, _ = frame.shape
    left_section = width // 3
    right_section = 2 * width // 3

    # Convert frame to RGB and perform inference
    results = model(frame, size=640)

    # Process each detection in the frame
    for *box, conf, cls in results.xyxy[0]:  # Accessing coordinates, confidence, and class ID
        conf = conf.item()
        cls = int(cls.item())
        
        if conf > 0.5:
            startX, startY, endX, endY = map(int, box)  # Get bounding box coordinates
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
            label = category_index[cls]
            label_text = f'{label}: {section}'
            cv2.putText(frame, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Debug feedback (optional)
            print(f"Detected {label} in the {section} section")

    # Draw section lines
    cv2.line(frame, (left_section, 0), (left_section, height), (0, 255, 0), 2)
    cv2.line(frame, (right_section, 0), (right_section, height), (0, 255, 0), 2)
    return frame


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_with_detections = detect_and_section(frame)

    # Show the processed frame (for debug purposes)
    cv2.imshow('YOLOR Real-time Detection', frame_with_detections)
    out.write(frame_with_detections)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()