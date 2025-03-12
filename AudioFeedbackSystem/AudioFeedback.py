import pyttsx3
import time
import queue

# Priority Weights (Higher = More Important)
OBJECT_TYPE_WEIGHTS = {
    "pedestrian": 5, "dog": 4, "car": 3, "bus": 3, "truck": 3, "motorcycle": 2,
    "bicycle": 2, "tricycle": 2, "reflective_cone": 2, "roadblock": 2, "fire_hydrant": 2,
    "pole": 2, "Trash_can": 1, "warning_column": 1, "sign": 1, "crosswalk": 3,
    "tactile_paving": 1, "tree": 1, "red_light": 1, "green_light": 1
}

LOCATION_WEIGHTS = {"left": 1, "middle": 2, "right": 1}

DISTANCE_PRIORITY = {
    "Very Close": 3,
    "Close": 2,
    "Far": 1  
}

def calculate_priority(obj):
    """Calculate priority based on distance, object type, and location."""
    distance_label = obj.get("distance", "very far")
    distance_factor = DISTANCE_PRIORITY.get(distance_label, 0)
    
    object_type_weight = OBJECT_TYPE_WEIGHTS.get(obj.get("label"), 0)
    location_weight = LOCATION_WEIGHTS.get(obj.get("location"), 0)
    
    return distance_factor + object_type_weight + location_weight

def generate_audio_feedback(queue):
    """Fetch the most recent detected objects and announce up to three important ones,
    filtering out any that are not 'Very Close' or 'Close'."""
    engine = pyttsx3.init()
    
    # Set voice.
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[1].id)  
    
    # Slow down the speaking speed.
    engine.setProperty("rate", 120)    
    engine.setProperty("volume", 1.0)  

    while True:
        latest_data = None
        # Flush the queue, keeping only the most recent message.
        while not queue.empty():
            latest_data = queue.get()
        
        if latest_data is None or "objects" not in latest_data or not latest_data["objects"]:
            time.sleep(0.1)
            continue

        # Use only the latest data from the most recent frame.
        objects = latest_data["objects"]
        # Filter to include only objects that are "Very Close" or "Close", ignore far objects
        objects = [obj for obj in objects if obj.get("distance") in ("Very Close", "Close")]
        if not objects:
            time.sleep(0.1)
            continue

        # Sort the objects by priority (highest first) and pick up to 3.
        sorted_objects = sorted(objects, key=calculate_priority, reverse=True)
        top_objects = sorted_objects[:3]

        summary_message_parts = []
        for obj in top_objects:
            label = obj['label'].replace('_', ' ')
            location_phrase = f"to the {obj['location']}" if obj['location'] != 'middle' else "in the middle"
            summary_message_parts.append(f"{label} at {obj['distance']} {location_phrase}")

        message = " and ".join(summary_message_parts)
        print("Speaking (Summary):", message)
        engine.say(message)
        engine.runAndWait()

        time.sleep(0.1)
