#pip install diffusers --upgrade
#pip install invisible_watermark transformers accelerate safetensors

from diffusers import DiffusionPipeline
import torch
from pathlib import Path
from slugify import slugify

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("mps")

prompt = "RAW photo, a ((Full Shot)) full naked body photo of 28 y.o beautiful ((Caucasian woman)) strong warrior in an (warrior princess) in ((cyberpunk)) (high detailed skin:1.2) 8k uhd, DSLR, soft lighting, high quality, film grain, Fujifilm XT3"
images = pipe(prompt=prompt).images
DIR_NAME="./images/"
dirpath = Path(DIR_NAME)
# create parent dir if doesn't exist
dirpath.mkdir(parents=True, exist_ok=True)

for idx, image in enumerate(images):
    image_name = f'{slugify(prompt)}-{idx}.png'
    image_path = dirpath / image_name
    image.save(image_path)
