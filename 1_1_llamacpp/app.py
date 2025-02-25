import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Chemin du cache local
CACHE_DIR = "./cache_model"
MODEL_NAME = "unsloth/Llama-3.2-1B"

def load_model():
    """ Charge le modèle et le tokenizer depuis le cache local """
    
    # S'assurer que le dossier cache existe
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Charger le modèle et le tokenizer en utilisant le cache
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, history=[], max_length=100):
    """ Génère un texte avec un historique de conversation """
    
    # Formater le prompt avec l'historique
    system_prompt = "Tu es un assistant IA utile et amical."
    formatted_prompt = system_prompt + "\n" + "\n".join(history) + "\n" + prompt

    # Tokenisation
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Génération du texte
    output = model.generate(**inputs, max_length=max_length)
    
    # Décodage de la réponse
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

if __name__ == "__main__":
    # Chargement du modèle
    model, tokenizer = load_model()

    # Historique de conversation
    history = [
        "Utilisateur: Quel est le rival principal de Naruto ?",
        "IA: Sasuke Uchiha."
    ]
    
    # Exemple de génération
    prompt = "Quel est le pouvoir principal de Luffy ?"
    response = generate_text(model, tokenizer, prompt, history)

    print("Réponse de l'IA:", response)