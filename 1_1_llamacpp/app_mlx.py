import os

from mlx_lm import load, generate
# Chargement du modèle MLX LLM
def load_model():
    model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

    return model , tokenizer

def format_prompt(system_prompt, history, user_input):
    """ Formate le prompt pour LLaMA 3.2 avec historique et prompt système """
    
    # Ajout du prompt système
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>\n\n"

    # Ajout de l'historique des échanges
    for role, text in history:
        role_tag = "user" if role == "user" else "assistant"
        prompt += f"<|start_header_id|>{role_tag}<|end_header_id|>\n{text}<|eot_id|>\n\n"

    # Ajout du message utilisateur
    prompt += f"<|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|>\n\n"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"  # Pour guider la génération

    return prompt

def generate_text(model,tokenizer, system_prompt, history, user_input, max_tokens=200):
    """ Génère une réponse en utilisant MLX LLM """
    
    # Formatter le prompt
    formatted_prompt = format_prompt(system_prompt, history, user_input)
    # Génération du texte
    output = generate(model, tokenizer,formatted_prompt, max_tokens=max_tokens)
    
    return output

if __name__ == "__main__":
    # Chargement du modèle
    model,tokenizer = load_model()

    # Définition du prompt système
    system_prompt = (
        "Tu es une IA qui répond uniquement en JSON valide. "
        "Ne donne aucune explication, aucune introduction, ni texte superflu. "
        "Réponds uniquement avec un objet JSON brut."
    )
    
    # Historique de conversation
    history = [
        ("user", "Quel est le rival principal de Naruto ?"),
        ("assistant", "{\"name\":\"sasuke\",\"power\":\"rinegan\",\"description\":\"Le Rinnegan (輪廻眼, signifiant littéralement : Œil de Samsâra) est réputé pour être le plus puissant des « Trois Grands Dôjutsu »,[3] les autres étant le Sharingan et le Byakugan. Il était dit qu'aux temps de troubles, celui qui manie le Rinnegan est envoyé des cieux pour devenir un « Dieu de Création » qui calmera le monde ou un « Dieu de la Destruction » qui réduira tout à néant. Le Rinnegan est caractérisé par un motif ondulatoire qui se propage sur tout le globe oculaire.[4] Le Rinnegan est caractérisé par des cercles concentriques ayant pour centre la pupille, l'iris et la sclérotique étant de couleur gris-violette\"}")
    ]

    # Exemple de génération
    user_input = "Quel est le pouvoir principal de Luffy ?"
    response = generate_text(model, tokenizer, system_prompt, history, user_input)

    print(response)