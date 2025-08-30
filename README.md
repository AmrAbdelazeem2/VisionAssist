# VisionAssist

**VisionAssist** is a mobile application that provides real-time object detection and depth estimation directly from a smartphone camera. Built using YOLO models and on-device TTS, it enables users to receive spoken feedback about nearby objects and their approximate distances.

## 🔑 Key Features

- **Real-Time Object Detection**  
  Utilizes a lightweight **YOLOv5 / YOLOv8** model optimized for mobile to detect objects in the live camera feed.

- **Depth Estimation**  
  Computes approximate distance-to-object using monocular depth inference models for spatial awareness.

- **Voice Narration**  
  Detected object classes and estimated distances are narrated using Android’s native **Text-to-Speech (TTS)** engine.

- **Tap-to-Capture**  
  Users can tap to capture a frame and receive an audio summary of objects and spatial layout in the scene.

---

## ⚙️ Tech Stack

| Component           | Technology Used            |
|---------------------|-----------------------------|
| Platform            | Android (Java / Kotlin)     |
| Object Detection    | YOLOv5 / YOLOv8 (TFLite)    |
| Depth Estimation    | MiDaS / Mobile Depth Model  |
| Audio Narration     | Android Text-to-Speech API  |
| IDE & Tools         | Android Studio, Gradle      |

---

## 📲 Usage

1. Launch the VisionAssist app on your Android device.
2. Grant camera and microphone permissions.
3. The live feed will start detecting and narrating objects with distance.
4. Tap the capture button to freeze a frame and receive a spoken summary.

---
