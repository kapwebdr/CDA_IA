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

# Demander le texte à analyser en français
texte = input("Entrez le texte en français à analyser : ")

results = classifier(texte)
print(results)

# Option pour analyser plusieurs textes
while True:
    more = input("Voulez-vous analyser un autre texte ? (o/n) : ")
    if more.lower() != 'o':
        break
    texte = input("Entrez le texte en français à analyser : ")
    results = classifier(texte)
    print(results)