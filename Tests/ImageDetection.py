from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
from pathlib import Path
import os
# Définir le chemin du cache
cache_dir = Path("../cache_model")
os.environ["HF_HOME"] = "../cache_model"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join("../cache_model", "hub")

# Créer le dossier cache s'il n'existe pas
cache_dir.mkdir(parents=True, exist_ok=True)

url = "https://cdn-imgix.headout.com/media/images/c90f7eb7a5825e6f5e57a5a62d05399c-25058-BestofParis-EiffelTower-Cruise-Louvre-002.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor       = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", cache_dir=cache_dir)
model           = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", cache_dir=cache_dir)

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
    )