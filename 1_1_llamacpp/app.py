import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Chemin du cache local
CACHE_DIR = "../cache_model"
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

def load_model():
    """ Charge le modèle et le tokenizer depuis le cache local """
    os.makedirs(CACHE_DIR, exist_ok=True)  # Création du dossier cache si nécessaire
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    return model, tokenizer

def format_prompt(system_prompt, history, user_input):
    """ Formate le prompt en respectant le format LLaMA 3.2 """

    # Ajout du prompt système
    prompt = f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>\n\n"

    # Ajout de l'historique des échanges
    for role, text in history:
        role_tag = "user" if role == "user" else "assistant"
        prompt += f"<|start_header_id|>{role_tag}<|end_header_id|>\n{text}<|eot_id|>\n\n"

    # Ajout du message de l'utilisateur actuel
    prompt += f"<|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|>\n\n"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"  # Pour guider la génération

    return prompt




def generate_text(model, tokenizer, system_prompt, history, user_input, max_length=200):
    """ Génère un texte basé sur un prompt formaté pour LLaMA 3.2 """
    
    # Formatter le prompt
    formatted_prompt = format_prompt(system_prompt, history, user_input)

    # Tokenisation
    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    # Génération du texte
    output = model.generate(**inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)

    # Décodage de la réponse
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

if __name__ == "__main__":
    # Chargement du modèle
    model, tokenizer = load_model()

    # Définition du prompt système
    system_prompt = "Tu es un assistant IA spécialisé en anime et manga."

    # Historique de conversation
    history = [
        ("user", "Quel est le rival principal de Naruto ?"),
        ("assistant", "Sasuke Uchiha.")
    ]
    
    # Exemple de génération
    user_input = "Quel est le pouvoir principal de Luffy ?"
    response = generate_text(model, tokenizer, system_prompt, history, user_input)

    print("Réponse de l'IA:", response)