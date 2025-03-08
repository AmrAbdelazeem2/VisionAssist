from transformers import pipeline
from PIL import Image


image_path = r"temp_images\frame_0003.jpg"
image = Image.open(image_path)

depth_estimator = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
result = depth_estimator(image)

depth_tensor = result["predicted_depth"]
depth_array = depth_tensor.detach().cpu().numpy()
print("Depth map shape:", depth_array.shape)
print("Sample depth values:")
print(depth_array[:5, :5])