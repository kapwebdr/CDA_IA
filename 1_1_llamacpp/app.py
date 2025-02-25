from transformers import AutoModelForCausalLM, AutoTokenizer

# Remplacez par le modèle Hugging Face correspondant (PyTorch)
MODEL_NAME = "unsloth/Llama-3.2-1B"

def load_model():
    """ Charge le modèle et le tokenizer depuis Hugging Face """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100):
    """ Génère un texte basé sur un prompt donné """
    inputs = tokenizer(prompt, return_tensors="pt")

    output = model.generate(**inputs, max_length=max_length)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

if __name__ == "__main__":
    # Chargement du modèle
    model, tokenizer = load_model()

    # Exemple de génération de texte
    prompt = "Écrivez un poème sur la nature"
    response = generate_text(model, tokenizer, prompt)

    print(response)