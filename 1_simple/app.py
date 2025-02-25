from transformers import pipeline

model = "distilgpt2"  # Modèle léger
text_generator = pipeline("text-generation", model=model)
result = text_generator("Bonjour, comment vas-tu ?", max_length=50)
print(result)