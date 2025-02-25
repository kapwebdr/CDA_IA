from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import json
import torch
from torch.utils.data import random_split

# 1. Charger un modèle plus grand et plus adapté
model_name = "bigscience/bloomz-1b1"  # Modèle multilingue plus grand
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True,  # Utilisation de la quantification 8-bit
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ajout des tokens spéciaux
special_tokens = {
    "pad_token": "[PAD]",
    "sep_token": "[SEP]",
    "additional_special_tokens": ["[QUESTION]", "[REPONSE]"]
}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Format plus structuré
        full_text = f"[QUESTION]{item['question']}[SEP][REPONSE]{item['answer']}[SEP]"
        
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Création d'un masque pour la perte
        labels = encodings["input_ids"].squeeze().clone()
        # Masquer les tokens qui ne font pas partie de la réponse
        question_part = tokenizer.encode(f"[QUESTION]{item['question']}[SEP][REPONSE]", add_special_tokens=False)
        labels[:len(question_part)] = -100
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": labels
        }

# Chargement et split du dataset
with open("dataset.json", "r", encoding="utf-8") as f:
    dataset_raw = json.load(f)

dataset = CustomDataset(dataset_raw, tokenizer)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Configuration LoRA optimisée
lora_config = LoraConfig(
    r=32,  # Rang plus élevé
    lora_alpha=64,
    target_modules=["query_key_value"],  # Modules spécifiques pour BLOOM
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
)

# Appliquer LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Configuration d'entraînement optimisée
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=100,  # Plus d'époques pour un petit dataset
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_ratio=0.1,
    logging_steps=1,
    save_steps=10,
    learning_rate=2e-4,
    fp16=True,  # Activation du mixed precision training
    remove_unused_columns=False,
    optim="paged_adamw_32bit",
    evaluation_strategy="steps",
    eval_steps=5,
    load_best_model_at_end=True,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Entraînement
trainer.train()

# Sauvegarder
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

def generate_response(question):
    # Format cohérent avec l'entraînement
    prompt = f"[QUESTION]{question}[SEP][REPONSE]"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            min_length=1,
            num_return_sequences=1,
            temperature=0.1,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    try:
        # Extraire uniquement la réponse
        response = response.split("[REPONSE]")[1].split("[SEP]")[0].strip()
        return response
    except:
        return "Erreur de génération"

# Mettre à jour requirements.txt avec les versions spécifiques