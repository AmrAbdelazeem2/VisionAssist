import cv2
from ultralytics import YOLO

# loading the pretrained yolo8nano model
model = YOLO('yolov8n.pt')


video_path = r'videoplayback.mp4'


cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('output_with_detections.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 

    #Running the model on the frame
    results = model(frame)

    # Render the detections on the frame
    result_frame = results[0].plot()

    # Write the frame with detections to the output video
    out.write(result_frame)

   #display the annotated frame
    cv2.imshow('YOLOv8 Video Detection', result_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. The video with detections has been saved as 'output_with_detections.mp4'.")