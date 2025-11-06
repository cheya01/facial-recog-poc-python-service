import json
from deepface import DeepFace
result = DeepFace.verify(img1_path="max_1.jpg", img2_path="max_2.webp",)
print(json.dumps(result, indent=2))