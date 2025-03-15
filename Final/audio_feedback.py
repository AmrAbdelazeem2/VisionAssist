import pyttsx3
import time
import queue

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
    """
    Fetch the latest detected objects from the queue and announce up to three
    important groups, grouping similar objects for a concise summary.
    """
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[1].id)
    engine.setProperty("rate", 120)
    engine.setProperty("volume", 1.0)

    while True:
        latest_data = None
        # Flush the queue, keeping only the most recent message.
        while not queue.empty():
            latest_data = queue.get()

        if not latest_data or "objects" not in latest_data or not latest_data["objects"]:
            time.sleep(0.1)
            continue

        # Filter objects for immediate feedback.
        objects = [obj for obj in latest_data["objects"] if obj.get("distance") in ("Very Close", "Close")]
        if not objects:
            time.sleep(0.1)
            continue

        groups = {}
        for obj in objects:
            key = (obj['label'], obj['distance'], obj['location'])
            priority = calculate_priority(obj)
            if key in groups:
                groups[key]['count'] += 1
                groups[key]['priority'] = max(groups[key]['priority'], priority)
            else:
                groups[key] = {'count': 1, 'priority': priority}

        sorted_groups = sorted(groups.items(), key=lambda x: x[1]['priority'], reverse=True)
        top_groups = sorted_groups[:3]

        summary_message_parts = []
        for (label, distance, location), info in top_groups:
            count = info['count']
            if count == 1:
                label_str = label.replace('_', ' ')
            elif count == 2:
                label_str = f"2 {label.replace('_', ' ')}s"
            else:
                label_str = f"multiple {label.replace('_', ' ')}s"
            distance_str = distance.lower()
            location_phrase = f"to your {location}" if location in ("left", "right") else "in the middle"
            summary_message_parts.append(f"{label_str} {distance_str} {location_phrase}")

        message = " and ".join(summary_message_parts)
        print("Speaking (Summary):", message)
        engine.say(message)
        engine.runAndWait()
        time.sleep(0.1)