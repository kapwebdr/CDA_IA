from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, random_split
import json
from torch.nn import CrossEntropyLoss

# 1. Charger le modèle et le tokenizer
# Utilisons un modèle plus adapté
model_name = "bigscience/bloom-560m"  # Plus grand modèle (560M paramètres)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ajout des tokens spéciaux
special_tokens = {"additional_special_tokens": ["<|question|>", "<|reponse|>", "<|fin|>"]}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

# 2. Préparer le dataset
class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Format plus clair pour l'apprentissage
        text = f"<|question|>{item['question']}<|reponse|>{item['answer']}<|fin|>"
        
        encodings = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()  # On garde les labels identiques aux inputs
        }

# 3. Créer le dataset
dataset = CustomDataset("dataset.json", tokenizer)

# Séparation en train et validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 4. Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir="./trained_model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=50,  # Plus d'époques car petit dataset
    save_steps=5,
    logging_steps=1,
    learning_rate=1e-5,   # Learning rate plus faible
    warmup_ratio=0.1,     # Warmup ratio au lieu de steps fixes
    weight_decay=0.01,    # Ajout de régularisation
    fp16=False,
    bf16=False,
    optim="adamw_torch",
    remove_unused_columns=False,
    seed=42,
    evaluation_strategy="steps",
    eval_steps=5,
    load_best_model_at_end=True,
    # Ajout de paramètres pour éviter le surapprentissage
    gradient_clipping=1.0,
    max_grad_norm=1.0,
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Masque pour se concentrer sur la partie réponse
        labels = inputs["labels"].clone()
        
        # Calcul de la perte uniquement sur la partie réponse
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# 5. Initialiser et lancer l'entraînement
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# 6. Sauvegarder le modèle
model_path = "./trained_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# 7. Fonction de test
def generate_text(prompt):
    formatted_prompt = f"<|question|>{prompt}<|reponse|>"
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
    
    outputs = model.generate(
        **inputs,
        max_length=64,        # Réduction de la longueur maximale
        min_length=1,         # Ajout d'une longueur minimale
        num_return_sequences=1,
        temperature=0.3,      # Température plus basse pour des réponses plus déterministes
        top_p=0.95,
        top_k=50,            # Limitation du vocabulaire aux 50 tokens les plus probables
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,  # Évite la répétition de trigrammes
        early_stopping=True
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)  # Gardons les tokens spéciaux
    try:
        response = generated_text.split("<|reponse|>")[1].split("<|fin|>")[0].strip()
        return response
    except:
        return "Erreur de génération"

# Test du modèle avec plus d'exemples
test_questions = [
    "Qui est le rival principal de Naruto ?",
    "Quel est le pouvoir principal de Luffy ?",
    "Dans quel anime Light Yagami possède un cahier magique ?",
    "Quelle est la capitale de la France ?"
]

print("\nTests du modèle entraîné:")
for question in test_questions:
    print(f"\nQuestion: {question}")
    print(f"Réponse: {generate_text(question)}")
