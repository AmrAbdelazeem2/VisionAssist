from ultralytics import YOLO
import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Normalize, Resize
import sys
import os
# Add the MiDaS directory to the Python path
sys.path.append(os.path.abspath('MiDaS'))
print(sys.path)
from midas.dpt_depth import DPTDepthModel
def prepare_image_for_inference(img):
    # Prepare the image for the model
    img = ToTensor()(img)
    img = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
    return img

# Load the YOLOv8 model (Change to yolov8m.pt for the medium version)
model = YOLO('best.pt')
category_index = model.names

# NOTE: change the path below for your corresponding video path
cap = cv2.VideoCapture('input_videos/university_walking.mp4')

# Get the original video's coordinates: width, height, and FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# video output writer
output_file = 'processed_videos/processed_output.mp4'  # Update this path for the desired output video name
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Load MiDaS model
def load_midas_model():
    model_path = "weights/dpt_large-midas-2f21e586.pt"  # Change to your MiDaS model file
    midas_model = DPTDepthModel(path=model_path, backbone="vitl16_384", non_negative=True)
    midas_model.eval()
    transform = Compose([
        Resize(384, 384),  # Adjust based on your model
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        prepare_image_for_inference,
    ])
    return midas_model, transform

midas_model, midas_transform = load_midas_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas_model.to(device)

# Process a single frame for depth estimation
def estimate_depth(frame, midas_model, midas_transform):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    img = midas_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = midas_model(img).squeeze().cpu().numpy()
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return depth

# Detect objects and overlay detections and depth map
def detect_and_section(frame):
    height, width, _ = frame.shape
    left_section = width // 3
    right_section = 2 * width // 3

    # Perform object detection on the frame
    results = model(frame)

    # Iterate the detected objects in the frame
    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()  # bounding box
        conf = result.conf.cpu().numpy()   # confidence score
        cls = int(result.cls.cpu().numpy())  # class ID

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

    # Estimate depth and combine it with the frame
    depth_map = estimate_depth(frame, midas_model, midas_transform)
    depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
    
    # Combine frame and depth map side by side
    combined = cv2.hconcat([frame, depth_colormap])
    return combined

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_with_detections = detect_and_section(frame)

    # Show the processed frame (NOTE: for debug and visualization purposes only)
    cv2.imshow('YOLOv8 with MiDaS Depth', frame_with_detections)

    out.write(frame_with_detections)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()