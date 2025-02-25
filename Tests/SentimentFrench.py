model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pathlib import Path
import os
# Définir le chemin du cache
cache_dir = Path("../cache_model")
# Créer le dossier cache s'il n'existe pas
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "../cache_model"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join("../cache_model", "hub")

model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
results = classifier("Tesla c'est génial")
print(results)