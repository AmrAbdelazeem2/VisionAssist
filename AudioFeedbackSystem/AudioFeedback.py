import pyttsx3
import time
import queue

# Priority Weights (Higher = More Important)
OBJECT_TYPE_WEIGHTS = {
    "pedestrian": 5, "dog": 4, "car": 3, "bus": 3, "truck": 3, "motorcycle": 2,
    "bicycle": 2, "tricycle": 2, "reflective_cone": 2, "roadblock": 2, "fire_hydrant": 2,
    "pole": 2, "Trash_can": 1, "warning_column": 1, "sign": 1, "crosswalk": 1,
    "tactile_paving": 1, "tree": 1, "red_light": 1, "green_light": 1
}

LOCATION_WEIGHTS = {"left": 1, "middle": 2, "right": 1}

DISTANCE_PRIORITY = {
    "immediate": 5,  
    "close": 4,
    "moderate": 3,
    "far": 2,
    "very far": 1  
}

def calculate_priority(obj):
    """Calculate priority based on distance, object type, and location."""
    distance_label = obj.get("distance", "very far")
    distance_factor = DISTANCE_PRIORITY.get(distance_label, 0)
    
    object_type_weight = OBJECT_TYPE_WEIGHTS.get(obj.get("label"), 0)
    location_weight = LOCATION_WEIGHTS.get(obj.get("location"), 0)
    
    return distance_factor + object_type_weight + location_weight

def generate_audio_feedback(queue):
    """Wait until new frame data is available, then process it."""
    engine = pyttsx3.init()

    # Set voice settings
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[1].id)  
    engine.setProperty("rate", 190)    
    engine.setProperty("volume", 1.0)  

    while True:
        print("Audio thread waiting for next frame data...")
        
        # Block and wait until a new frame's objects arrive
        latest_data = queue.get()  

        if "objects" not in latest_data or not latest_data["objects"]:
            continue  # Skip empty frames

        print("Received frame data for audio feedback.")

        # Sort objects by priority (highest first)
        sorted_objects = sorted(latest_data["objects"], key=calculate_priority, reverse=True)

        for obj in sorted_objects:
            label = obj['label'].replace('_', ' ')
            location_phrase = "in the middle" if obj['location'] == 'middle' else f"to the {obj['location']}"
            message = f"{label} detected {obj['distance']} {location_phrase}."
            
            print("Speaking:", message)
            engine.say(message)
            engine.runAndWait()
            time.sleep(0.5)  # Small delay between object announcements

        print("Audio feedback complete. Waiting for next frame...")