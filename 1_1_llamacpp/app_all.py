import argparse
import os
import time
from typing import Tuple, Optional

CACHE_DIR = "../cache_model"
SYSTEM_PROMPTS_DIR = "./system_prompts"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SYSTEM_PROMPTS_DIR, exist_ok=True)

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

def setup_model(model_type: str) -> Tuple[any, Optional[any]]:
    """Configure et retourne le modèle selon le type choisi"""
    if model_type == "transformers":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = "unsloth/Llama-3.2-1B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR)
        return model, tokenizer
    
    elif model_type == "llamacpp":
        from llama_cpp import Llama
        model_path = os.path.join(CACHE_DIR, "llama-3.2-1b-instruct.Q4_K_M.gguf")
        if not os.path.exists(model_path):
            llm = Llama.from_pretrained(
                repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
                filename="*Q4_K_M.gguf",
                cache_dir=CACHE_DIR,
                verbose=False
            )
        else:
            llm = Llama(model_path=model_path)
        return llm, None
    
    elif model_type == "mlx":
        os.environ["HF_HOME"] = CACHE_DIR
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")
        
        from mlx_lm import load
        model, tokenizer = load(
            "mlx-community/Llama-3.2-3B-Instruct-4bit"
        )
        return model, tokenizer

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
    """Génère une réponse selon le type de modèle et retourne la réponse, le temps et les tokens/s"""
    formatted_prompt = format_prompt(prompt, system_prompt, history)
    start_time = time.time()
    
    if model_type == "transformers":
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        input_tokens = len(inputs.input_ids[0])
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_tokens = len(outputs[0])
    
    elif model_type == "llamacpp":
        # Tokenization du prompt d'entrée
        input_tokens = model.tokenize(formatted_prompt.encode())
        n_input_tokens = len(input_tokens)
        
        # Génération de la réponse
        response = model(
            formatted_prompt,  # Pas besoin d'encoder ici, l'API le fait
            max_tokens=4096,
            temperature=0.7,
            top_p=0.95,
            repeat_penalty=1.15
        )
        response = response['choices'][0]['text'].strip()
        
        # Tokenization de la sortie
        output_tokens = model.tokenize(response.encode())
        n_output_tokens = len(output_tokens)
        
        # Calcul du nombre de tokens générés
        tokens_generated = n_output_tokens - n_input_tokens
    
    elif model_type == "mlx":
        from mlx_lm import generate
        input_tokens = len(tokenizer.encode(formatted_prompt))
        output = generate(
            model,
            tokenizer,
            formatted_prompt,
            max_tokens=200,
        )
        response = output
        output_tokens = len(tokenizer.encode(output))

    end_time = time.time()
    duration = end_time - start_time
    tokens_per_second = tokens_generated / duration if 'tokens_generated' in locals() else (output_tokens - input_tokens) / duration

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
        model, tokenizer = setup_model(model_type)
        
        # Sélection du system prompt après le chargement du modèle
        system_prompt = select_system_prompt()

        # Initialisation de l'historique
        history = []
        
        # Boucle de conversation
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
            
            # Ajout à l'historique
            history.append(("user", prompt))
            history.append(("assistant", response))
            
            # Limiter l'historique aux 10 derniers échanges (5 tours de conversation)
            if len(history) > 10:
                history = history[-10:]
            
            print("=== Réponse ===")
            print(response)
            print("==============")
            print(f"\nTemps de génération : {duration:.2f} secondes")
            print(f"Vitesse : {tokens_per_second:.2f} tokens/s")
            
            # Afficher l'historique de conversation
            # print("\n=== Historique de la conversation ===")
            # for i, (role, text) in enumerate(history):
            #     speaker = "Vous" if role == "user" else "Assistant"
            #     print(f"\n{speaker}: {text}")
    
    except Exception as e:
        print(f"\nErreur lors de l'exécution: {str(e)}")

if __name__ == "__main__":
    main() 