import multiprocessing
import random
import time

# List of all objects the model detects
DETECTED_OBJECTS = [
    "tree", "red_light", "green_light", "crosswalk", "tactile_paving", "sign", 
    "pedestrian", "bicycle", "bus", "truck", "car", "motorcycle", "reflective_cone", 
    "Trash_can", "warning_column", "roadblock", "pole", "dog", "tricycle", "fire_hydrant"
]

LOCATIONS = ["left", "middle", "right"]

DISTANCES = ["immediate", "close", "moderate", "far", "very far"] #  "Very Close", "Close", "Far"

def generate_random_objects():
    """Generates a random list of detected objects with different distances and locations."""
    num_objects = random.randint(1, 5)  # Random number of detected objects per cycle
    detected_data = {"objects": []}

    for _ in range(num_objects):
        obj = {
            "label": random.choice(DETECTED_OBJECTS),
            "distance": random.choice(DISTANCES),  # Random distance between 1m - 10m
            "location": random.choice(LOCATIONS)
        }
        detected_data["objects"].append(obj)
    
    print("Detected Objects: \n" + str(detected_data))
    return detected_data

def detect_objects(queue):
    """Simulated object detection module with dynamic random outputs."""
    while True:
        detected_data = generate_random_objects()
        queue.put(detected_data)  # Send data to audio process
        time.sleep(random.uniform(1.5, 3.5))  # Simulate variable detection timing

if __name__ == "__main__":
    queue = multiprocessing.Queue()
    detect_objects(queue)  # Run only if directly executed
