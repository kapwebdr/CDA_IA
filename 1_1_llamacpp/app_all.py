import os
import time
from typing import Tuple, Optional
from model_utils import ModelConfig, ModelLoader, ModelGenerator

CACHE_DIR = "../cache_model"
SYSTEM_PROMPTS_DIR = "./system_prompts"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SYSTEM_PROMPTS_DIR, exist_ok=True)

MODEL_CONFIGS = {
    "transformers": ModelConfig(
        model_type="transformers",
        model_path="unsloth/Llama-3.2-1B-Instruct"
    ),
    "llamacpp": ModelConfig(
        model_type="llamacpp",
        model_path="bartowski/Llama-3.2-1B-Instruct-GGUF"
    ),
    "mlx": ModelConfig(
        model_type="mlx",
        model_path="mlx-community/Llama-3.2-3B-Instruct-4bit"
    )
}

def get_system_prompts() -> dict:
    """Récupère tous les system prompts disponibles dans le dossier"""
    prompts = {}
    if not os.listdir(SYSTEM_PROMPTS_DIR):
        # Créer un prompt par défaut si le dossier est vide
        default_prompt = "Tu es un assistant IA serviable et honnête. Tu réponds toujours de manière concise et précise."
        with open(os.path.join(SYSTEM_PROMPTS_DIR, "default.txt"), "w") as f:
            f.write(default_prompt)
    
    for filename in os.listdir(SYSTEM_PROMPTS_DIR):
        if filename.endswith('.txt'):
            with open(os.path.join(SYSTEM_PROMPTS_DIR, filename), 'r') as f:
                prompts[filename[:-4]] = f.read().strip()
    return prompts

def select_system_prompt() -> str:
    """Interface pour sélectionner un system prompt"""
    prompts = get_system_prompts()
    print("\nSystem prompts disponibles:")
    for idx, (name, _) in enumerate(prompts.items(), 1):
        print(f"{idx}. {name}")
    
    while True:
        try:
            choice = int(input("\nChoisissez un system prompt (numéro) : "))
            if 1 <= choice <= len(prompts):
                selected_prompt = list(prompts.values())[choice-1]
                return selected_prompt
        except ValueError:
            pass
        print("Choix invalide. Veuillez entrer un numéro valide.")

def format_prompt(prompt: str, system_prompt: str, history=None) -> str:
    """Formate le prompt selon le format LLaMA 3.2"""
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>\n\n"

    if history:
        for role, text in history:
            role_tag = "user" if role == "user" else "assistant"
            formatted_prompt += f"<|start_header_id|>{role_tag}<|end_header_id|>\n{text}<|eot_id|>\n\n"

    formatted_prompt += f"<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|>\n\n"
    formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n"

    return formatted_prompt

def generate_response(model_type: str, model, tokenizer, prompt: str, system_prompt: str, history=None) -> Tuple[str, float, float]:
    """Génère une réponse selon le type de modèle"""
    formatted_prompt = format_prompt(prompt, system_prompt, history)
    start_time = time.time()
    
    config = MODEL_CONFIGS[model_type]
    
    if model_type == "transformers":
        response, input_tokens, output_tokens = ModelGenerator.generate_transformers(
            model, tokenizer, formatted_prompt, config
        )
    elif model_type == "llamacpp":
        response, input_tokens, output_tokens = ModelGenerator.generate_llamacpp(
            model, formatted_prompt, config
        )
    elif model_type == "mlx":
        response, input_tokens, output_tokens = ModelGenerator.generate_mlx(
            model, tokenizer, formatted_prompt, config
        )

    end_time = time.time()
    duration = end_time - start_time
    tokens_per_second = (output_tokens - input_tokens) / duration

    return response, duration, tokens_per_second

def main():
    print("\n=== Sélecteur de Modèle IA ===\n")
    print("Modèles disponibles :")
    print("1. transformers")
    print("2. llamacpp")
    print("3. mlx")
    
    while True:
        choice = input("\nChoisissez votre modèle (1-3) : ")
        if choice in ['1', '2', '3']:
            model_types = {
                '1': 'transformers',
                '2': 'llamacpp',
                '3': 'mlx'
            }
            model_type = model_types[choice]
            break
        print("Choix invalide. Veuillez choisir un nombre entre 1 et 3.")

    try:
        print("\nChargement du modèle...")
        config = MODEL_CONFIGS[model_type]
        loader = ModelLoader()
        
        if model_type == "transformers":
            model, tokenizer = loader.load_transformers(config)
        elif model_type == "llamacpp":
            model, tokenizer = loader.load_llamacpp(config)
        elif model_type == "mlx":
            model, tokenizer = loader.load_mlx(config)
        
        system_prompt = select_system_prompt()
        history = []
        
        while True:
            print("\nEntrez votre message (ou 'quit' pour quitter) :")
            lines = []
            while True:
                line = input()
                if line.lower() == 'quit':
                    print("\nAu revoir!")
                    return
                if line:
                    lines.append(line)
                elif lines:
                    break
            prompt = '\n'.join(lines)
            
            print("\nGénération de la réponse...\n")
            response, duration, tokens_per_second = generate_response(
                model_type, model, tokenizer, prompt, system_prompt, history
            )
            
            history.append(("user", prompt))
            history.append(("assistant", response))
            
            if len(history) > 10:
                history = history[-10:]
            
            print("=== Réponse ===")
            print(response)
            print("==============")
            print(f"\nTemps de génération : {duration:.2f} secondes")
            print(f"Vitesse : {tokens_per_second:.2f} tokens/s")
    
    except Exception as e:
        print(f"\nErreur lors de l'exécution: {str(e)}")

if __name__ == "__main__":
    main() 