# pip install --upgrade pip
# pip install --upgrade transformers scipy

from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy
from pathlib import Path
from slugify import slugify

DIR_NAME="./audio/"
dirpath = Path(DIR_NAME)
# create parent dir if doesn't exist
dirpath.mkdir(parents=True, exist_ok=True)

processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large")

prompt = "Dub step wooble synth"

inputs = processor(
    text=[prompt],
    padding=True,
    return_tensors="pt",
)

audio_values = model.generate(**inputs, max_new_tokens=256)

sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write(dirpath / f'{slugify(prompt)}-.wav', rate=sampling_rate, data=audio_values[0, 0].numpy())
