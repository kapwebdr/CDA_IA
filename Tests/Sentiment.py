#python -m venv .env
#source .env/bin/activate
#pip install transformers
#pip install torch torchvision torchaudio
#python Sentiment.py

from transformers import pipeline

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