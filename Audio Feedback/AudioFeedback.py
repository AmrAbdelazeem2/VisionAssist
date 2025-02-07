import multiprocessing
import pyttsx3

# Priority Weights (Higher = More Important)
OBJECT_TYPE_WEIGHTS = {
    "pedestrian": 5, "dog": 4, "car": 3, "bus": 3, "truck": 3, "motorcycle": 2,
    "bicycle": 2, "tricycle": 2, "reflective_cone": 2, "roadblock": 2, "fire_hydrant": 2,
    "pole": 2, "Trash_can": 1, "warning_column": 1, "sign": 1, "crosswalk": 1,
    "tactile_paving": 1, "tree": 1, "red_light": 1, "green_light": 1
}
LOCATION_WEIGHTS = {"left": 1, "middle": 2, "right": 1}

def calculate_priority(obj):
    """Calculate priority based on distance, object type, and location."""
    distance_factor = 1 / obj["distance"]  # Closer objects have higher priority
    object_type_weight = OBJECT_TYPE_WEIGHTS.get(obj["label"], 0)
    location_weight = LOCATION_WEIGHTS.get(obj["location"], 0)
    
    return distance_factor + object_type_weight + location_weight


def generate_audio_feedback(queue):
    """Fetch the latest detected objects and announce the highest priority one."""
    engine = pyttsx3.init()
    
    # 🎤 Set Voice (Change Index Based on Your Preferred Voice)
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[1].id)  # Change index to your preferred voice
    engine.setProperty("rate", 190)  # Speed (default ~200, lower = slower)
    engine.setProperty("volume", 1.0)  # Max volume (0.0 to 1.0)


    last_spoken_object = None  # Avoid repeating the same object

    latest_data = None  # Ensure latest_data is always defined

    while True:
        # Retrieve the latest available data (clear old detections)
        while not queue.empty():
            latest_data = queue.get()  # Get the latest available data

        # Ensure latest_data exists before accessing it
        if latest_data is None or "objects" not in latest_data or not latest_data["objects"]:
            continue  # Skip if no valid detection

        # Select object with the highest priority
        highest_priority_obj = max(latest_data["objects"], key=calculate_priority)

        # Avoid repeating the same object
        if highest_priority_obj["label"] == last_spoken_object:
            continue

        # Generate speech output
        message = f"{highest_priority_obj['label']} detected at {highest_priority_obj['distance']} meters on the {highest_priority_obj['location']}."
        print("🎤 Speaking (Priority Object):", message)
        
        engine.say(message)
        engine.runAndWait()  # Speak the message
        
        last_spoken_object = highest_priority_obj["label"]  # Save last spoken object


if __name__ == "__main__":
    queue = multiprocessing.Queue()
    generate_audio_feedback(queue)  # Run only if directly executed
