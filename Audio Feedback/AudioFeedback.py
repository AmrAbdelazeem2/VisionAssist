import pyttsx3
import time

# Priority Weights (Higher = More Important)
OBJECT_TYPE_WEIGHTS = {
    "pedestrian": 5, "dog": 4, "car": 3, "bus": 3, "truck": 3, "motorcycle": 2,
    "bicycle": 2, "tricycle": 2, "reflective_cone": 2, "roadblock": 2, "fire_hydrant": 2,
    "pole": 2, "Trash_can": 1, "warning_column": 1, "sign": 1, "crosswalk": 1,
    "tactile_paving": 1, "tree": 1, "red_light": 1, "green_light": 1
}

LOCATION_WEIGHTS = {"left": 1, "middle": 2, "right": 1}

DISTANCE_PRIORITY = {
    "immediate": 5,  # Highest priority for the closest objects
    "close": 4,
    "moderate": 3,
    "far": 2,
    "very far": 1  # Lowest priority for the most distant objects
}


def calculate_priority(obj):
    """Calculate priority based on distance, object type, and location."""
    distance_label = obj.get("distance", "very far")
    distance_factor = DISTANCE_PRIORITY.get(distance_label, 0)
    
    # Retrieve weights for object type and location.
    object_type_weight = OBJECT_TYPE_WEIGHTS.get(obj.get("label"), 0)
    location_weight = LOCATION_WEIGHTS.get(obj.get("location"), 0)
    
    # The final priority is a sum modal of the weights
    return distance_factor + object_type_weight + location_weight


def generate_audio_feedback(queue):
    """Fetch the latest detected objects and announce the highest priority one."""
    engine = pyttsx3.init()
    
    # Set voice.
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[1].id)  
    
    engine.setProperty("rate", 190)    
    engine.setProperty("volume", 1.0)  

    last_spoken_object = None  
    latest_data = None         

    while True:
        # Retrieve the latest available data by emptying the queue.
        while not queue.empty():
            latest_data = queue.get()
            # print("New data received:", latest_data)

        # If there's no valid data, wait a short time before re-checking.
        if latest_data is None or "objects" not in latest_data or not latest_data["objects"]:
            time.sleep(0.1)
            continue

        # choose the object with the highest priority.
        highest_priority_obj = max(latest_data["objects"], key=calculate_priority)
        
        # If the object is the same as last time optionally wait before speaking again
        if highest_priority_obj["label"] == last_spoken_object:
            time.sleep(0.1)
            continue

        # Replace underscores with spaces in the label.
        label = highest_priority_obj['label'].replace('_', ' ')

        # Construct the speech message using the modified label.
        if highest_priority_obj['location'] == 'middle':
            location_phrase = "in the middle"
        else:
            location_phrase = f"to the {highest_priority_obj['location']}"

        message = (
            f"{label} was detected "
            f"{highest_priority_obj['distance']} "
            f"{location_phrase}."
        )

        print("Speaking (Priority Object):", message)
        
        # Queue the message for speech and wait until speaking is complete.
        engine.say(message)
        engine.runAndWait()
        
        # Update last spoken object.
        last_spoken_object = highest_priority_obj["label"]
        
        # Small delay to reduce CPU usage.
        time.sleep(0.1)
