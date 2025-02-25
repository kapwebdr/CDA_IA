from diffusers import DiffusionPipeline
import torch
from pathlib import Path
from slugify import slugify
import os

# Définir le chemin du cache
cache_dir = Path("../cache_model")
os.environ["HF_HOME"] = "../cache_model"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join("../cache_model", "hub")

# Créer le dossier cache s'il n'existe pas
cache_dir.mkdir(parents=True, exist_ok=True)

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    cache_dir=cache_dir
)
pipe.to("mps")

# Demander le prompt à l'utilisateur
prompt = input("Entrez votre description d'image (prompt) : ")

images = pipe(prompt=prompt).images
DIR_NAME="./images/"
dirpath = Path(DIR_NAME)
# create parent dir if doesn't exist
dirpath.mkdir(parents=True, exist_ok=True)

for idx, image in enumerate(images):
    # Utiliser str() pour s'assurer que le prompt est une chaîne de caractères
    image_name = f'{slugify(str(prompt))}-{idx}.png'
    image_path = dirpath / image_name
    image.save(image_path)
