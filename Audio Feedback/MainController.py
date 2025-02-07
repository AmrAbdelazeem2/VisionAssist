import multiprocessing
from ObjectDetection import detect_objects
from AudioFeedback import generate_audio_feedback

if __name__ == "__main__":
    queue = multiprocessing.Queue()

    # Start Object Detection Process
    detection_process = multiprocessing.Process(target=detect_objects, args=(queue,))
    detection_process.start()

    # Start Audio Feedback Process
    audio_process = multiprocessing.Process(target=generate_audio_feedback, args=(queue,))
    audio_process.start()

    # Keep the main script running
    detection_process.join()
    audio_process.join()
