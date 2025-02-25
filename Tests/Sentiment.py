from transformers import pipeline
from pathlib import Path
import os
# Définir le chemin du cache
cache_dir = Path("../cache_model")
# Créer le dossier cache s'il n'existe pas
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "../cache_model"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join("../cache_model", "hub")

classifier = pipeline("sentiment-analysis")
classifier("We are very happy to show you the 🤗 Transformers library.")
#[{'label': 'POSITIVE', 'score': 0.9998}]

#If you have more than one input, pass your inputs as a list to the pipeline() to return a list of dictionaries:

results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

# label: POSITIVE, with score: 0.9998
# label: NEGATIVE, with score: 0.5309
# ARCHFLAGS="-arch arm64" pip3 install numpy  --compile --no-cache-dir --force-reinstall

# Demander le texte à analyser
text = input("Entrez le texte à analyser : ")

result = classifier(text)
print(f"label: {result[0]['label']}, with score: {round(result[0]['score'], 4)}")

# Option pour analyser plusieurs textes
while True:
    more = input("Voulez-vous analyser un autre texte ? (o/n) : ")
    if more.lower() != 'o':
        break
    text = input("Entrez le texte à analyser : ")
    result = classifier(text)
    print(f"label: {result[0]['label']}, with score: {round(result[0]['score'], 4)}")