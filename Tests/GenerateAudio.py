from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy
from pathlib import Path
from slugify import slugify
import os
# Définir le chemin du cache
cache_dir = Path("../cache_model")
# Créer le dossier cache s'il n'existe pas
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "../cache_model"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join("../cache_model", "hub")

DIR_NAME="./audio/"
dirpath = Path(DIR_NAME)
# create parent dir if doesn't exist
dirpath.mkdir(parents=True, exist_ok=True)

processor = AutoProcessor.from_pretrained("facebook/musicgen-large", cache_dir=cache_dir)
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large", cache_dir=cache_dir)

# Demander le prompt à l'utilisateur
prompt = input("Entrez votre description audio (prompt) : ")

inputs = processor(
    text=[prompt],
    padding=True,
    return_tensors="pt",
)

audio_values = model.generate(**inputs, max_new_tokens=256)

sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write(dirpath / f'{slugify(prompt)}-.wav', rate=sampling_rate, data=audio_values[0, 0].numpy())
